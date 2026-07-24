import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# pixel_unshuffle rearranges (N, C, H, W) -> (N, C*r*r, H/r, W/r) by folding each
# r*r spatial block into the channel dim. It is a CONTIGUOUS-STORE + COMPUTED-INDEX
# GATHER-LOAD (out is written linearly; the input is read at a permuted position).
#
# The GENERIC ops/pixel_unshuffle.py raw @triton.jit kernel has NO @libentry cache, so
# on XPU each launch re-enters the compile/launch path -> the IR dump
# (harness/perf_ir_3/ir-pixel_unshuffle-dev3.log) shows ~2752 kernel modules (838K
# lines) and the tiny benchmark shapes measure 150-190ms (speedup ~0.000). Routing the
# same gather kernel through the tuned pointwise_dynamic copy instead REGRESSED the
# large shape (6D StridedBuffer strided-gather codegen ~1.6ms vs the raw gather 0.1ms).
#
# Fix: keep the raw contiguous-store / gather-load kernel (it is already the right
# access pattern for the large shape) but add @libentry caching + a size-banded
# BLOCK/num_warps launch (values passed EXPLICITLY, never via heuristics). This kills
# the per-launch recompile catastrophe on the tiny shapes while preserving the fast
# gather on the large shape. Kernel body unchanged (zero correctness risk).


@libentry()
@triton.jit
def pixel_unshuffle_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    N,
    C,
    H,
    W,
    R,
    C_out,
    H_out,
    W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    sN_in = C * H * W
    sC_in = H * W
    sH_in = W

    sN_out = C_out * H_out * W_out
    sC_out = H_out * W_out
    sH_out = W_out

    n = offsets // sN_out
    rem1 = offsets - n * sN_out
    c_out = rem1 // sC_out
    rem2 = rem1 - c_out * sC_out
    h_out = rem2 // sH_out
    w_out = rem2 - h_out * sH_out

    r2 = R * R
    c_in = c_out // r2
    remc = c_out - c_in * r2
    dh = remc // R
    dw = remc - dh * R

    h_in = h_out * R + dh
    w_in = w_out * R + dw

    in_index = n * sN_in + c_in * sC_in + h_in * sH_in + w_in

    x = tl.load(in_ptr + in_index, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _launch_config(n_elements):
    # size-banded BLOCK / num_warps (explicit, no heuristics)
    if n_elements <= 1024:
        return 256, 2
    elif n_elements <= 8192:
        return 1024, 4
    elif n_elements <= 65536:
        return 4096, 8
    return 16384, 8


def _launch(inp, r, out):
    N, C, H, W = inp.shape
    C_out = C * r * r
    H_out = H // r
    W_out = W // r
    n_elements = inp.numel()
    if n_elements == 0:
        return
    BLOCK_SIZE, num_warps = _launch_config(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(inp.device):
        pixel_unshuffle_kernel[grid](
            inp,
            out,
            n_elements,
            N,
            C,
            H,
            W,
            r,
            C_out,
            H_out,
            W_out,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )


def pixel_unshuffle(input, downscale_factor, *, layout=None):
    logger.debug("GEMS_KUNLUNXIN PIXEL_UNSHUFFLE")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
    N, C, H, W = x.shape
    r = int(downscale_factor)
    assert r > 0, "downscale_factor must be > 0"
    assert (H % r == 0) and (
        W % r == 0
    ), "H and W must be divisible by downscale_factor"

    out_shape = (N, C * r * r, H // r, W // r)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    _launch(x, r, out)
    return out


def pixel_unshuffle_out(input, downscale_factor, out):
    logger.debug("GEMS_KUNLUNXIN PIXEL_UNSHUFFLE_OUT")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
    N, C, H, W = x.shape
    r = int(downscale_factor)
    assert r > 0, "downscale_factor must be > 0"
    assert (H % r == 0) and (
        W % r == 0
    ), "H and W must be divisible by downscale_factor"
    expected_shape = (N, C * r * r, H // r, W // r)
    assert out.shape == expected_shape, f"out must have shape {expected_shape}"
    assert out.dtype == x.dtype, "out dtype must match input dtype"
    assert out.device == x.device, "out device must match input device"
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")

    _launch(x, r, out)
    return out
