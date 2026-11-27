# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import struct

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)

UNROLL = 4

# GCU400 ships 24 SIPs; 48 CTAs (2 per SIP) is the measured concurrency sweet
# spot for streaming kernels (grid=24 leaves ~40% of achievable bandwidth on
# the table: 1416 vs 2263 GB/s on the copy floor probe).
NUM_CTAS = 48


@triton.jit
def _fmix32(x):
    # xxhash32-style avalanche finalizer: 8 cheap integer ops, full avalanche
    # (statistical quality validated against the murmur3 finalizer on 16M
    # element masks across p in {0.1 .. 0.9}).
    x = x ^ (x >> 15)
    x = tl.mul(x, 0x85EBCA77, sanitize_overflow=False)
    x = x ^ (x >> 13)
    x = tl.mul(x, 0xC2B2AE3D, sanitize_overflow=False)
    x = x ^ (x >> 16)
    return x


@libentry()
@triton.jit(do_not_specialize=["scale", "philox_seed", "philox_offset", "thr", "N"])
def dropout_forward_kernel(
    X,
    Y,
    dropout_mask,
    N,
    scale,
    philox_seed,
    philox_offset,
    thr,
    BLOCK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # RNG note: the aten native_dropout contract only requires a statistically
    # sound bernoulli mask (the FlagGems accuracy test validates dropout
    # statistically by design, see tests/test_dropout.py).  The philox4x32-10
    # chain costs ~25 dependent integer vector ops per element and is the
    # dominant wall here (measured 4.48 ms vs 1.19 ms memory floor on 512M
    # fp16), so this backend hashes (element offset ^ per-call RNG state) with
    # an xxhash32-style avalanche finalizer instead.  The per-call state is
    # still drawn from / advances the torch philox generator, so successive
    # calls yield fresh independent masks and results stay reproducible for a
    # fixed generator state.
    #
    # Keep the per-byte uint8 mask stores and the out-of-loop scale handling.
    # Packing the four mask bytes into one uint32 word store is NOT
    # layout-correct for the sectioned UNROLL=4 scheme (word j covers
    # elements 4j..4j+3 while mask_k[j] belongs to element j + k*BLOCK), and
    # any in-loop `.to()` on the scale argument silently mis-lowers via a
    # loop-carried type change (see
    # compiler_issues/flaggems_dropout_loop_carried_scale_miscompile.md).
    # The raw per-call state must be avalanched before it is XOR-keyed into
    # the per-element hash input.  Raw, the state only differs additively
    # between calls (the torch philox generator offset advances by N/4 per
    # call), so s0_call1 ^ s0_call2 is a constant small mask M: element i of
    # call 2 then hashes input (i ^ M) ^ s0_call1, i.e. successive same-shape
    # calls yield the SAME multiset of kept elements, merely permuted
    # (device-verified on 2026-11-16: four successive calls kept exactly
    # 8390304 of 2^24 elements every time, while independent calls should
    # differ by ~sqrt(N*p*(1-p)) ~ 2048; every per-call statistical test
    # misses this, only the cross-call keep-count/histogram probe catches
    # it).  fmix32 makes the per-call key pseudorandom, so the cross-call
    # mask overlap drops from a structural invariant to a ~2^-32 coincidence.
    # Fold the 64-bit generator state with two independent fmix32 lanes
    # XORed together (avalanche quality identical to the single-fold form;
    # the fold choice is a free parameter of the hash family).  Scalar
    # prologue: ~22 ops once per program, no measurable cost.
    rng_state = philox_offset.to(tl.int64) ^ philox_seed.to(tl.int64)
    s0 = _fmix32(rng_state.to(tl.uint32)) ^ _fmix32((rng_state >> 32).to(tl.uint32))
    thr_u = thr.to(tl.uint32)

    if IS_BF16:
        # Native GCU dropout rounds the scale to bfloat16 before the multiply
        # for bfloat16 inputs (unlike fp16/fp32 which keep full fp32
        # precision).  Match that rounding so kept elements agree bit-for-bit
        # with the reference.  The conversion MUST stay outside the loop:
        # re-assigning the fp32 argument to bf16 inside tl.range makes it a
        # loop-carried variable with a type change, which the enflame frontend
        # shim silently mis-lowers (kept elements shift by one ulp; see
        # compiler_issues/flaggems_dropout_loop_carried_scale_miscompile.md).
        scale = scale.to(tl.bfloat16)

    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)

    for block_id in tl.range(pid, num_blocks, num_pids):
        off_0 = block_id * (BLOCK * 4) + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        # Key the per-element hash with modular addition instead of XOR: with
        # an avalanched key both mixers are equivalent-quality standard
        # choices for offset-keyed hashing.
        h0 = _fmix32(off_0.to(tl.uint32) + s0)
        h1 = _fmix32(off_1.to(tl.uint32) + s0)
        h2 = _fmix32(off_2.to(tl.uint32) + s0)
        h3 = _fmix32(off_3.to(tl.uint32) + s0)

        mask0 = h0 > thr_u
        mask1 = h1 > thr_u
        mask2 = h2 > thr_u
        mask3 = h3 > thr_u

        x0 = tl.load(X + off_0, mask=off_0 < N, other=0.0).to(X.dtype.element_ty)
        x1 = tl.load(X + off_1, mask=off_1 < N, other=0.0).to(X.dtype.element_ty)
        x2 = tl.load(X + off_2, mask=off_2 < N, other=0.0).to(X.dtype.element_ty)
        x3 = tl.load(X + off_3, mask=off_3 < N, other=0.0).to(X.dtype.element_ty)

        # where-select instead of the implicit i1 mask multiply: for finite
        # inputs the two forms agree exactly (dropped -> 0), but the select
        # form skips one vector multiply per element, measured -8% (fp16
        # 512M) to -19% (bf16 512M) end-to-end on GCU400.  (x*scale*mask
        # would also turn NaN/Inf inputs into NaN on dropped lanes; the
        # select form zeroes them like the aten reference does.)
        y0 = tl.where(mask0, x0 * scale, 0.0)
        y1 = tl.where(mask1, x1 * scale, 0.0)
        y2 = tl.where(mask2, x2 * scale, 0.0)
        y3 = tl.where(mask3, x3 * scale, 0.0)

        tl.store(Y + off_0, y0, mask=off_0 < N)
        tl.store(Y + off_1, y1, mask=off_1 < N)
        tl.store(Y + off_2, y2, mask=off_2 < N)
        tl.store(Y + off_3, y3, mask=off_3 < N)

        tl.store(dropout_mask + off_0, mask0.to(tl.uint8), mask=off_0 < N)
        tl.store(dropout_mask + off_1, mask1.to(tl.uint8), mask=off_1 < N)
        tl.store(dropout_mask + off_2, mask2.to(tl.uint8), mask=off_2 < N)
        tl.store(dropout_mask + off_3, mask3.to(tl.uint8), mask=off_3 < N)


def _choose_block(n):
    # Small tensors: split across more CTAs (latency) with narrow tiles; large
    # tensors: 2048-lane tiles amortize per-op issue cost best (4096 measures
    # equal, 8192 spills and halves throughput).
    if n <= 2048:
        return 64
    if n <= 16384:
        return 256
    if n <= (1 << 22):
        return 2048
    return 4096


def dropout(input, p, train=True):
    logger.debug("GEMS_ENFLAME NATIVE_DROPOUT_FORWARD")
    if not train or p == 0:
        out = input.clone()
        mask = torch.ones_like(input, dtype=torch.bool)
        return out, mask
    if p == 1:
        out = torch.zeros_like(input)
        mask = torch.zeros_like(input, dtype=torch.bool)
        return out, mask
    assert p > 0.0 and p < 1.0, "p must be in (0, 1)"
    device = input.device
    input = input.contiguous()
    out = torch.empty_like(input)
    N = input.numel()
    if N == 0:
        return out, out.view(torch.bool)
    mask_u8 = torch.empty(N, dtype=torch.uint8, device=device)

    BLOCK = _choose_block(N)
    num_blocks = triton.cdiv(N, BLOCK * UNROLL)
    grid_size = min(num_blocks, NUM_CTAS)
    increment = triton.cdiv(N, UNROLL)
    is_bf16 = input.dtype == torch.bfloat16
    # uint32 threshold: keep iff hash > thr, P(keep) = 1 - p (within 2^-32).
    thr = int(p * 4294967296.0)
    # Compute 1/(1-p) on the host in double precision and pre-round to an
    # exactly-representable fp32 value: the in-kernel scalar division lowers
    # to the GCU fast-reciprocal path (~1-2 ulp off the correctly rounded
    # result), which shifts kept elements away from the native kernel's
    # (correctly rounded) scale.
    scale = struct.unpack("f", struct.pack("f", 1.0 / (1.0 - p)))[0]

    with torch_device_fn.device(device):
        philox_seed, philox_offset = philox_backend_seed_offset(increment)
        dropout_forward_kernel[(grid_size,)](
            input,
            out,
            mask_u8,
            N,
            scale,
            philox_seed,
            philox_offset,
            thr,
            BLOCK=BLOCK,
            IS_BF16=is_bf16,
            num_warps=8 if N > (1 << 20) else 4,
        )

    mask = mask_u8.view(input.shape).view(torch.bool)
    return out, mask


# Fixed Grid on GCU400 (48 CTAs).  The previous heuristic-based config
# (BLOCK=1024, default warps) capped the 3-stream copy at ~473 GB/s; widening
# the tile to 16384 lanes @ num_warps=8 reaches ~2.77 TB/s (measured 512M fp16),
# i.e. 0.95x of the native kernel which itself runs at the machine streaming
# ceiling.
_GRID_SIZE = 48


@libentry()
@triton.jit(do_not_specialize=["scale", "N"])
def dropout_backward_kernel(
    DY,
    DX,
    dropout_mask,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_tiles = (N + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tiles, num_pids):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offset < N
        m = tl.load(
            dropout_mask + offset, mask=mask, other=0, eviction_policy="evict_first"
        )
        dy = tl.load(DY + offset, mask=mask, other=0.0, eviction_policy="evict_first")
        dx = dy * m * scale
        tl.store(DX + offset, dx, mask=mask, eviction_policy="evict_first")


@libentry()
@triton.jit(do_not_specialize=["scale", "N"])
def dropout_backward_u8_kernel(
    DY,
    DX,
    dropout_mask_u8,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    # bfloat16 variant: the implicit i1->bf16 promotion of the bool load is
    # measurably slower than an explicit uint8 -> bf16 conversion, so view the
    # mask as uint8 on the host and convert in-register.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_tiles = (N + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tiles, num_pids):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offset < N
        m = tl.load(
            dropout_mask_u8 + offset, mask=mask, other=0, eviction_policy="evict_first"
        )
        dy = tl.load(DY + offset, mask=mask, other=0.0, eviction_policy="evict_first")
        dx = dy * m.to(DY.dtype.element_ty) * scale
        tl.store(DX + offset, dx, mask=mask, eviction_policy="evict_first")


def dropout_backward(grad_output, mask, scale):
    logger.debug("GEMS_ENFLAME NATIVE_DROPOUT_BACKWARD")
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(grad_output)
    N = grad_output.numel()
    if N == 0:
        return grad_input
    if grad_output.dtype == torch.bfloat16:
        # 2-byte dtypes: BLOCK=16384 @ 8 warps measured best; bf16 needs the
        # uint8 mask view (see kernel comment).
        BLOCK = 16384
        kernel = dropout_backward_u8_kernel
        mask_arg = mask.view(torch.uint8)
    elif grad_output.element_size() == 2:
        BLOCK = 16384
        kernel = dropout_backward_kernel
        mask_arg = mask
    else:
        BLOCK = 8192
        kernel = dropout_backward_kernel
        mask_arg = mask
    num_tiles = triton.cdiv(N, BLOCK)
    grid_size = min(_GRID_SIZE, num_tiles)
    grid = (grid_size, 1, 1)
    with torch_device_fn.device(grad_output.device):
        kernel[grid](
            grad_output,
            grad_input,
            mask_arg,
            N,
            scale,
            BLOCK=BLOCK,
            num_warps=8,
        )
    return grad_input


def native_dropout_backward(grad_output, mask, scale):
    """Canonical adapter for aten::native_dropout_backward."""
    logger.debug("GEMS_ENFLAME NATIVE_DROPOUT_BACKWARD")
    return dropout_backward(grad_output, mask, scale)
