import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# narrow_copy (kunlunxin / XPU override).
#
# The generic ops/narrow_copy.py kernel (source loc narrow_copy.py:12) has two
# XPU pathologies:
#   1. NO @libentry -> the kernel is recompiled per shape (IR explosion, the
#      112K-line dump in ir-narrow_copy-dev2.log; same class as sgn_/cat_out).
#   2. THREE per-element integer div/mod to decode the flat output index
#      (idx // (length*dim_prod_post), (idx // dim_prod_post) % length,
#      idx % dim_prod_post). XPU has no hardware divider -> these choke
#      throughput to ~8-15 GB/s (vs torch ~1700 GB/s) even though the op is a
#      pure memory copy.
#
# Structural fact: narrowing dim `d` of a CONTIGUOUS input [pre, dim_size, post]
# (pre = prod(shape[:d]), post = prod(shape[d+1:])) produces `pre` blocks, each
# `length*post` elements. Within a block the output slice out[p, :, :] and the
# input slice inp[p, start:start+length, :] are BOTH fully contiguous
# (consecutive l map to consecutive inp positions). Only ACROSS p is there a
# jump (inp advances dim_size*post, out advances length*post).
#
# So a 2D grid (p, chunk-of-block) needs ZERO div/mod: program (p, j) copies a
# contiguous BLOCK-run at out[p*block_out + j*BLOCK ...] from
# inp[p*inp_block + base + j*BLOCK ...]. Single store per program (only the last
# chunk is partially masked -> safe on XPU). For dim==0, pre==1 and this is a
# plain contiguous memcpy with a base offset.


@libentry()
@triton.jit
def _narrow_copy_flat_kernel(
    out_ptr,
    inp_ptr,
    base,  # start * post (uniform pointer shift into input)
    N,  # total elements to copy (= pre==1 -> length*post)
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # pre == 1 fast path (dim==0 or all leading dims size-1): a PURE contiguous
    # memcpy. off = pid*BLOCK + arange uses a compile-time stride (BLOCK is
    # constexpr) and `base` is a uniform pointer shift, so OffsetAnalysis proves
    # stride-1 -> block DMA. (A 2D grid with p*block_out puts a runtime-stride *
    # program_id term in the per-lane index and defeats that proof -> discrete.)
    #
    # A single store + partial mask (only the last chunk is masked) is SAFE on
    # XPU. An UNROLL=8 bulk/tail split was measured SLOWER here (0.37 vs 0.75
    # avg): the extra tail kernel + wide-block launch tanks the small benchmark
    # shapes while the large shapes barely move. NOTE 2026-08-12: on the large
    # shapes a BLOCK=16384 tile + UNMASKED path (NEED_MASK=0 when N % BLOCK == 0)
    # is required anyway: at BLOCK=8192 the fp16/bf16 kernels only reach
    # ~910 GB/s on (1024,65536), while BLOCK=16384 unmasked reaches ~1620 GB/s
    # (1.77x). Masked-vs-unmasked at BOTH 8192 and 16384 is identical for fp32
    # and ~3% apart for fp16/bf16, so the mask cost alone is not the issue —
    # the tile size is.
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = off < N
        x = tl.load(inp_ptr + base + off, mask=mask)
        tl.store(out_ptr + off, x, mask=mask)
    else:
        x = tl.load(inp_ptr + base + off)
        tl.store(out_ptr + off, x)


@libentry()
@triton.jit
def _narrow_copy_blocked_kernel(
    out_ptr,
    inp_ptr,
    block_out,  # length * post  (contiguous run per pre-block)
    inp_block,  # dim_size * post (input stride between pre-blocks)
    base,  # start * post        (offset into each input block)
    BLOCK: tl.constexpr,
):
    # pre > 1 general path (narrowing a non-leading dim): `pre` contiguous blocks.
    # 2D grid (p, chunk) -> no div/mod. The p*block_out / p*inp_block terms are
    # runtime-stride * program_id (discrete on XPU), but this path is only hit by
    # the correctness tests (the benchmark narrows dim 0 -> pre==1 fast path).
    p = tl.program_id(0)
    within = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = within < block_out
    out_off = p * block_out + within
    inp_off = p * inp_block + base + within
    x = tl.load(inp_ptr + inp_off, mask=mask)
    tl.store(out_ptr + out_off, x, mask=mask)


def _copy_block(block_out):
    b = triton.next_power_of_2(block_out)
    if b > 16384:
        b = 16384
    if b < 256:
        b = 256
    return b


def narrow_copy(inp, dim, start, length):
    logger.debug("GEMS_KUNLUNXIN NARROW_COPY")
    assert (
        dim >= -inp.ndim and dim < inp.ndim
    ), f"Invalid dim: {dim} for tensor with {inp.ndim} dimensions"
    dim = dim % inp.ndim

    if start < 0:
        start = start % inp.size(dim)
    length = min(length, inp.size(dim) - start)

    out_shape = list(inp.shape)
    out_shape[dim] = length
    # Allocate via torch.empty_strided (NOT registered by gems) instead of
    # torch.empty: inside a gems context torch.empty dispatches to the gems
    # `empty` override, which (a) runs a zeroing triton store kernel over the
    # whole buffer we are about to overwrite anyway and (b) recompiles that
    # kernel on every call in this environment (~100ms/call JIT storm), ~3x
    # ending up in the measured latency. empty_strided is a pure native
    # allocation and is functionally identical for the contiguous output.
    stride = [1] * len(out_shape)
    for i in range(len(out_shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * out_shape[i + 1]
    out = torch.empty_strided(out_shape, stride, dtype=inp.dtype, device=inp.device)

    if out.numel() == 0:
        return out

    inp = inp.contiguous()

    dim_size = inp.size(dim)
    post = 1
    for d in range(dim + 1, inp.ndim):
        post *= inp.size(d)
    pre = 1
    for d in range(dim):
        pre *= inp.size(d)

    block_out = length * post
    inp_block = dim_size * post
    base = start * post

    BLOCK = _copy_block(block_out)
    num_warps = 16 if BLOCK >= 2048 else (8 if BLOCK >= 512 else 4)

    if pre == 1:
        # dim==0 (or all leading dims size-1): a PURE contiguous memcpy. Use the
        # 1D flat kernel (pid*BLOCK compile-time stride + uniform `base` pointer
        # shift -> stride-1 block DMA). This is the benchmark hot path.
        grid = (triton.cdiv(block_out, BLOCK),)
        _narrow_copy_flat_kernel[grid](
            out,
            inp,
            base,
            block_out,
            BLOCK=BLOCK,
            NEED_MASK=block_out % BLOCK != 0,
            num_warps=num_warps,
        )
    else:
        # Narrowing a non-leading dim: `pre` contiguous blocks -> 2D grid, no
        # div/mod (correctness path for tests).
        grid = (pre, triton.cdiv(block_out, BLOCK))
        _narrow_copy_blocked_kernel[grid](
            out,
            inp,
            block_out,
            inp_block,
            base,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return out
