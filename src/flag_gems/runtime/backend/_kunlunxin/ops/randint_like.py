import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    uint_to_uniform_float,
)

from .rand import choose_unroll

logger = logging.getLogger(__name__)

# randint_like fills the output with random integers in [0, high). The GENERIC
# ops/randint_like.py kernel is decorated with @triton.heuristics(get_heuristic_config("rand"))
# which supplies BLOCK at LAUNCH via a grid=lambda. On the XPU triton fork this
# "heuristic-supplied launch param" path re-enters the compile/launch path per launch ->
# the IR dump (ir-randint_like-dev7.log) explodes to ~13.9M lines.
#
# Fix (same recipe as rand_/bernoulli_): DROP @triton.heuristics; compute UNROLL / BLOCK /
# grid EXPLICITLY in the Python wrapper (choose_unroll from rand.py) and pass them to the
# kernel. Two kernels mirror rand_kernel_1 (UNROLL<=4, single philox value) and
# rand_kernel_2 (UNROLL up to 16), each scaling the uniform float by `high` and truncating
# to int. Kernel algorithm is identical to the generic one (zero correctness risk); the
# test only checks the output range [0, high).


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randint_kernel_1(
    out_ptr,
    N,
    high,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, _r1, _r2, _r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    high_f = high.to(tl.float32)
    i0 = (r0 * high_f).to(tl.int32)
    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    tl.store(out_ptr + off_0, i0, mask=off_0 < N, eviction_policy="evict_first")


@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def randint_kernel_2(
    out_ptr,
    N,
    high,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r4, r5, r6, r7 = tl.philox(philox_seed, c0 + 1, c1, _O, _O)
    r8, r9, r10, r11 = tl.philox(philox_seed, c0 + 2, c1, _O, _O)
    r12, r13, r14, r15 = tl.philox(philox_seed, c0 + 3, c1, _O, _O)
    high_f = high.to(tl.float32)
    i0 = (uint_to_uniform_float(r0) * high_f).to(tl.int32)
    i1 = (uint_to_uniform_float(r1) * high_f).to(tl.int32)
    i2 = (uint_to_uniform_float(r2) * high_f).to(tl.int32)
    i3 = (uint_to_uniform_float(r3) * high_f).to(tl.int32)
    i4_ = (uint_to_uniform_float(r4) * high_f).to(tl.int32)
    i5 = (uint_to_uniform_float(r5) * high_f).to(tl.int32)
    i6 = (uint_to_uniform_float(r6) * high_f).to(tl.int32)
    i7 = (uint_to_uniform_float(r7) * high_f).to(tl.int32)
    i8 = (uint_to_uniform_float(r8) * high_f).to(tl.int32)
    i9 = (uint_to_uniform_float(r9) * high_f).to(tl.int32)
    i10 = (uint_to_uniform_float(r10) * high_f).to(tl.int32)
    i11 = (uint_to_uniform_float(r11) * high_f).to(tl.int32)
    i12 = (uint_to_uniform_float(r12) * high_f).to(tl.int32)
    i13 = (uint_to_uniform_float(r13) * high_f).to(tl.int32)
    i14 = (uint_to_uniform_float(r14) * high_f).to(tl.int32)
    i15 = (uint_to_uniform_float(r15) * high_f).to(tl.int32)
    off_0 = tl.program_id(0) * BLOCK * UNROLL + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    off_4 = off_3 + BLOCK
    off_5 = off_4 + BLOCK
    off_6 = off_5 + BLOCK
    off_7 = off_6 + BLOCK
    off_8 = off_7 + BLOCK
    off_9 = off_8 + BLOCK
    off_10 = off_9 + BLOCK
    off_11 = off_10 + BLOCK
    off_12 = off_11 + BLOCK
    off_13 = off_12 + BLOCK
    off_14 = off_13 + BLOCK
    off_15 = off_14 + BLOCK
    tl.store(out_ptr + off_0, i0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, i1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, i2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, i3, mask=off_3 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_4, i4_, mask=off_4 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_5, i5, mask=off_5 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_6, i6, mask=off_6 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_7, i7, mask=off_7 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_8, i8, mask=off_8 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_9, i9, mask=off_9 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_10, i10, mask=off_10 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_11, i11, mask=off_11 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_12, i12, mask=off_12 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_13, i13, mask=off_13 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_14, i14, mask=off_14 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_15, i15, mask=off_15 < N, eviction_policy="evict_first")


def randint_like(
    self,
    high,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    logger.debug("GEMS_KUNLUNXIN RANDINT_LIKE")
    if device is None:
        device = self.device
    if dtype is None:
        dtype = self.dtype
    out = torch.empty_like(self, device=device, dtype=dtype)
    N = self.numel()
    if N == 0:
        return out
    cluster_num = 12
    UNROLL = choose_unroll(N)
    BLOCK_SIZE = min(triton.next_power_of_2(triton.cdiv(N, cluster_num * UNROLL)), 1024)
    grid_fn = triton.cdiv(N, BLOCK_SIZE * UNROLL)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(increment)
    with torch_device_fn.device(self.device):
        if UNROLL <= 4:
            randint_kernel_1[(grid_fn,)](
                out, N, high, philox_seed, philox_offset, BLOCK_SIZE, UNROLL
            )
        else:
            randint_kernel_2[(grid_fn,)](
                out, N, high, philox_seed, philox_offset, BLOCK_SIZE, UNROLL
            )
    return out
