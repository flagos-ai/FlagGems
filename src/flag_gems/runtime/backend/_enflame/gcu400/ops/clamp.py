# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24
_FLAT_GRID_CAP = NUM_SIPS * 2

# --------------------------------------------------------------------------
# pointwise_dynamic fallback path (non-contiguous / non-float / broadcast /
# mixed-dtype bounds).  Keeps the exact generic semantics of the
# distribution implementation for every case the flat kernels below do not
# take over.
# --------------------------------------------------------------------------


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def clamp_func_tensor(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_min_tensor(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_max_tensor(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def clamp_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_min(x, mini):
    return tl.maximum(mini, x.to(tl.float32))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_max(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


# --------------------------------------------------------------------------
# GCU400 flat grid-stride kernels.  The generic pointwise_dynamic path
# dispatches a block-pointer kernel whose per-CTA in-flight tile (65536
# elements x 4 streams x ~12B/elem ~= 786KB) sits above the ~448KB DSM
# concurrency threshold, capping mid/large-shape bandwidth at
# 1.88-2.47TB/s where the 4-stream flat kernel reaches 2.66TB/s.  Ladders
# below were swept with do_bench(median, kernel mode) on GCU400
# (BLOCK x num_warps, fp16/fp32, 4K/16M/256M/1G); the scalar-bound ladder
# matches the validated clamp_max/sigmoid/special_expit one (same 2-stream
# kernel shape).  fp32 tensor-bound >2^28 keeps the pointwise path: the
# flat variant measured 6546us vs 6427us there (no flat win).
# --------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N_total", "mini", "maxi"])
def clamp_scalar_flat_kernel(
    x_ptr,
    out_ptr,
    mini,
    maxi,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    HAS_MIN: tl.constexpr,
    HAS_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off).to(tl.float32)
            if HAS_MIN and HAS_MAX:
                v = tl.minimum(maxi, tl.maximum(mini, x))
            elif HAS_MIN:
                v = tl.maximum(mini, x)
            else:
                v = tl.minimum(maxi, x)
            tl.store(out_ptr + off, v.to(out_ptr.dtype.element_ty))
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
            if HAS_MIN and HAS_MAX:
                v = tl.minimum(maxi, tl.maximum(mini, x))
            elif HAS_MIN:
                v = tl.maximum(mini, x)
            else:
                v = tl.minimum(maxi, x)
            tl.store(out_ptr + off, v.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def clamp_tensor_flat_kernel(
    x_ptr,
    lo_ptr,
    hi_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off).to(tl.float32)
            lo = tl.load(lo_ptr + off).to(tl.float32)
            hi = tl.load(hi_ptr + off).to(tl.float32)
            v = tl.minimum(hi, tl.maximum(lo, x))
            tl.store(out_ptr + off, v.to(out_ptr.dtype.element_ty))
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
            lo = tl.load(lo_ptr + off, mask=mask).to(tl.float32)
            hi = tl.load(hi_ptr + off, mask=mask).to(tl.float32)
            v = tl.minimum(hi, tl.maximum(lo, x))
            tl.store(out_ptr + off, v.to(out_ptr.dtype.element_ty), mask=mask)


def _choose_block_tensor(n_total, itemsize):
    # 4-stream sweep (GCU400, do_bench median, kernel mode):
    #   16M fp16 B65536/nw2 60.2us (2.23TB/s) vs pointwise 71.5us;
    #   16M fp32 B32768/nw4 105.6us vs pointwise 113.6us;
    #   1G  fp16 B65536/nw2 3233us (2.66TB/s) vs pointwise 3442us.
    # Returns (block, num_warps) or (0, 0) when the flat path has no win:
    #   launch-floor shapes (N<=8192) measured 3.30us on the pointwise path
    #   vs 3.59us flat (64,64) core case; fp32 tensor-bound >2^28 measured
    #   6546us flat vs 6427us pointwise.
    if n_total <= 8192:
        return 0, 0
    if n_total <= 65536:
        return 8192, 2
    if itemsize <= 2:
        return 65536, 2
    if n_total <= (1 << 28):
        return 32768, 4
    return 0, 0


def _choose_block_scalar(n_total, itemsize):
    # Validated 2-stream ladder shared by clamp_max/sigmoid/special_expit.
    if n_total <= 8192:
        return max(256, triton.next_power_of_2(n_total)), 2
    if n_total <= 65536:
        return 8192, 2
    if n_total <= 524288:
        return 32768, 2
    if n_total > (1 << 28) and itemsize <= 2:
        return 131072, 2
    return 65536, 2


def _flat_eligible(A):
    return A.is_contiguous() and A.dtype.is_floating_point


def _tensor_bound_full(A, bound):
    return (
        isinstance(bound, torch.Tensor)
        and bound.dtype == A.dtype
        and bound.shape == A.shape
        and bound.is_contiguous()
    )


def clamp(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLAMP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    if (
        _flat_eligible(A)
        and not isinstance(mini, torch.Tensor)
        and not isinstance(maxi, torch.Tensor)
    ):
        out = torch.empty_like(A)
        n_total = A.numel()
        if n_total == 0:
            return out
        block, num_warps = _choose_block_scalar(n_total, A.element_size())
        grid = (min(triton.cdiv(n_total, block), _FLAT_GRID_CAP),)
        with torch_device_fn.device(A.device):
            clamp_scalar_flat_kernel[grid](
                A,
                out,
                mini if mini is not None else 0.0,
                maxi if maxi is not None else 0.0,
                n_total,
                BLOCK=block,
                EVEN=n_total % block == 0,
                HAS_MIN=mini is not None,
                HAS_MAX=maxi is not None,
                num_warps=num_warps,
            )
        return out
    if mini is None:
        return clamp_func_max(A, maxi)
    if maxi is None:
        return clamp_func_min(A, mini)
    return clamp_func(A, mini, maxi)


def clamp_(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLAMP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    if (
        _flat_eligible(A)
        and not isinstance(mini, torch.Tensor)
        and not isinstance(maxi, torch.Tensor)
    ):
        n_total = A.numel()
        if n_total == 0:
            return A
        block, num_warps = _choose_block_scalar(n_total, A.element_size())
        grid = (min(triton.cdiv(n_total, block), _FLAT_GRID_CAP),)
        with torch_device_fn.device(A.device):
            clamp_scalar_flat_kernel[grid](
                A,
                A,
                mini if mini is not None else 0.0,
                maxi if maxi is not None else 0.0,
                n_total,
                BLOCK=block,
                EVEN=n_total % block == 0,
                HAS_MIN=mini is not None,
                HAS_MAX=maxi is not None,
                num_warps=num_warps,
            )
        return A
    if mini is None:
        return clamp_func_max(A, maxi, out0=A)
    if maxi is None:
        return clamp_func_min(A, mini, out0=A)
    return clamp_func(A, mini, maxi, out0=A)


def clamp_tensor(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLAMP_TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    if (
        _flat_eligible(A)
        and _tensor_bound_full(A, mini)
        and _tensor_bound_full(A, maxi)
    ):
        out = torch.empty_like(A)
        n_total = A.numel()
        if n_total == 0:
            return out
        block, num_warps = _choose_block_tensor(n_total, A.element_size())
        if block:
            grid = (min(triton.cdiv(n_total, block), _FLAT_GRID_CAP),)
            with torch_device_fn.device(A.device):
                clamp_tensor_flat_kernel[grid](
                    A,
                    mini,
                    maxi,
                    out,
                    n_total,
                    BLOCK=block,
                    EVEN=n_total % block == 0,
                    num_warps=num_warps,
                )
            return out
        return clamp_func_tensor(A, mini, maxi)
    if mini is None:
        return clamp_func_max_tensor(A, maxi)
    if maxi is None:
        return clamp_func_min_tensor(A, mini)
    return clamp_func_tensor(A, mini, maxi)


def clamp_tensor_(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLAMP_TENSOR_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    if (
        _flat_eligible(A)
        and _tensor_bound_full(A, mini)
        and _tensor_bound_full(A, maxi)
    ):
        n_total = A.numel()
        if n_total == 0:
            return A
        block, num_warps = _choose_block_tensor(n_total, A.element_size())
        if block:
            grid = (min(triton.cdiv(n_total, block), _FLAT_GRID_CAP),)
            with torch_device_fn.device(A.device):
                clamp_tensor_flat_kernel[grid](
                    A,
                    mini,
                    maxi,
                    A,
                    n_total,
                    BLOCK=block,
                    EVEN=n_total % block == 0,
                    num_warps=num_warps,
                )
            return A
        return clamp_func_tensor(A, mini, maxi, out0=A)
    if mini is None:
        return clamp_func_max_tensor(A, maxi, out0=A)
    if maxi is None:
        return clamp_func_min_tensor(A, mini, out0=A)
    return clamp_func_tensor(A, mini, maxi, out0=A)
