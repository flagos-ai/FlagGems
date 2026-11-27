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
from flag_gems.utils import libentry, pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24
_FLAT_GRID_CAP = NUM_SIPS * 2
# int32 flat-offset safety bound: element offsets must stay < 2**31
_FLAT_MAX_ELEMS = 1 << 31


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


def clamp_tensor(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLAMP_TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max_tensor(A, maxi)
    elif maxi is None:
        return clamp_func_min_tensor(A, mini)
    else:
        return clamp_func_tensor(A, mini, maxi)


def clamp_tensor_(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLAMP__TENSOR")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max_tensor(A, maxi, out0=A)
    elif maxi is None:
        return clamp_func_min_tensor(A, mini, out0=A)
    else:
        return clamp_func_tensor(A, mini, maxi, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def clamp_func(x, mini, maxi):
    return tl.minimum(maxi, tl.maximum(mini, x.to(tl.float32)))


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_min(x, mini):
    return tl.maximum(x.to(tl.float32), mini)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def clamp_func_max(x, maxi):
    return tl.minimum(maxi, x.to(tl.float32))


# --------------------------------------------------------------------------
# GCU400 flat grid-stride kernel for tensor-bound clamp_min (3 streams:
# x load + min load + store).  The generic pointwise_dynamic broadcast path
# costs 2.37-2.75TB/s effective bandwidth where this flat kernel reaches
# 2.46-2.89TB/s (do_bench median w1000/r100, GCU400/L600), and
# torch.clamp_min(tensor, tensor) natively runs a single fused kernel at the
# same DRAM floor, so the win comes from saturating it more closely.
#
# BODY=2 keeps the aten promote-to-f32 math (load->f32 cvt->maximum->cvt
# back).  BODY=3 compares in the tensor's own 16-bit domain; for fp16 the two
# are bit-identical on a 262873-item battery (NaN payload, +-0 sign,
# subnormals, non-representable values) because 16-bit values promote to f32
# exactly and rounding back is monotone, and skipping both cvt chains wins
# ~2% on fp16 mid shapes.  bf16/fp32 stay on BODY=2 (bf16 native-domain
# maximum measured no win; fp32 promotion is a no-op).
# --------------------------------------------------------------------------


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def clamp_min_flat_kernel(
    x_ptr,
    mini_ptr,
    out_ptr,
    N_total,
    BODY: tl.constexpr,
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
            if BODY == 2:
                x = tl.load(x_ptr + off).to(tl.float32)
                m = tl.load(mini_ptr + off).to(tl.float32)
                tl.store(out_ptr + off, tl.maximum(m, x).to(out_ptr.dtype.element_ty))
            else:
                x = tl.load(x_ptr + off)
                m = tl.load(mini_ptr + off)
                tl.store(out_ptr + off, tl.maximum(m, x))
        else:
            mask = off < N_total
            if BODY == 2:
                x = tl.load(x_ptr + off, mask=mask).to(tl.float32)
                m = tl.load(mini_ptr + off, mask=mask).to(tl.float32)
                tl.store(
                    out_ptr + off,
                    tl.maximum(m, x).to(out_ptr.dtype.element_ty),
                    mask=mask,
                )
            else:
                x = tl.load(x_ptr + off, mask=mask)
                m = tl.load(mini_ptr + off, mask=mask)
                tl.store(out_ptr + off, tl.maximum(m, x), mask=mask)


def _choose_cfg(n_total, dtype, inplace):
    # (BLOCK, num_warps, BODY, grid_cap) ladder, swept with do_bench median
    # (w1000/r100, kernel mode) on GCU400/L600, interleaved 3-round A/B:
    # - fp16 B65536/nw2/BODY=3 above the pointwise bands: 2^24 40.78us vs
    #   native 41.61; 2^30 2230.70us (cap48) / 2204.85us (cap96, oop) vs
    #   native 2443.96.  cap96 only pays off above 2^26 and only measured
    #   out-of-place, in-place keeps the validated cap48.
    # - bf16 B65536/nw2/BODY=2: 2^28 546.95 vs 583.30; 2^30 2191.82 vs
    #   2314.93 (B131072/nw4 lost the same-window re-run).
    # - fp32 B32768/nw2/BODY=2: 2^28 1135.62 vs 1221.95; 2^30 4463.82 vs
    #   4881.25.
    # - launch-floor shapes (N<=8192) 3.17-3.30us vs native 4.7us.
    if dtype == torch.float16:
        body = 3
    else:
        body = 2
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 256), 2, body, _FLAT_GRID_CAP
    if n_total <= 65536:
        return 8192, 2, body, _FLAT_GRID_CAP
    if n_total <= 524288:
        return 32768, 2, body, _FLAT_GRID_CAP
    if dtype == torch.float16:
        cap = _FLAT_GRID_CAP
        if not inplace and n_total > (1 << 26):
            cap = _FLAT_GRID_CAP * 2
        return 65536, 2, 3, cap
    if dtype == torch.bfloat16:
        return 65536, 2, 2, _FLAT_GRID_CAP
    return 32768, 2, 2, _FLAT_GRID_CAP


# Host-level flake guard: on this shared multi-tenant host the extern
# toolchain (gcu_compiler_llvm_ir_converter.sh and even plain rm/mkdir/flock
# children) intermittently segfaults during a fresh compile under fork-storm
# load; the same launch then succeeds on retry (failed compiles are not
# cached).  Only signature-matched toolchain flakes are retried; deterministic
# errors raise.  Same markers as the production clamp_max/sigmoid kernels.
_TOOLCHAIN_FLAKE_MARKERS = (
    "Segmentation fault",
    "converterResult",
    "serializing the module",
    "core dumped",
)


def _launch_flat(args, grid, block, n_total, num_warps, body):
    for attempt in range(3):
        try:
            clamp_min_flat_kernel[grid](
                *args,
                n_total,
                BODY=body,
                BLOCK=block,
                EVEN=n_total % block == 0,
                num_warps=num_warps,
            )
            return
        except Exception as exc:
            if attempt < 2 and any(m in str(exc) for m in _TOOLCHAIN_FLAKE_MARKERS):
                logger.debug(
                    "clamp_min toolchain flake on attempt %d, retrying", attempt
                )
                continue
            raise


def _flat_eligible(A, mini):
    return (
        A.is_contiguous()
        and A.dtype.is_floating_point
        and isinstance(mini, torch.Tensor)
        and mini.dtype == A.dtype
        and mini.shape == A.shape
        and mini.is_contiguous()
        and mini.device == A.device
        and A.numel() < _FLAT_MAX_ELEMS
    )


def clamp_min(A, mini):
    if mini is None:
        raise ValueError("Mini must not be None")
    if _flat_eligible(A, mini):
        out = torch.empty_like(A)
        n_total = A.numel()
        if n_total == 0:
            return out
        block, num_warps, body, cap = _choose_cfg(n_total, A.dtype, False)
        grid = (min(triton.cdiv(n_total, block), cap),)
        with torch_device_fn.device(A.device):
            _launch_flat((A, mini, out), grid, block, n_total, num_warps, body)
        return out
    if isinstance(mini, torch.Tensor):
        if mini.device != A.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, "
                f"but found at least two devices, {A.device} and {mini.device}"
            )
        return clamp_func_min_tensor(A, mini)
    return clamp_func_min(A, mini)


def clamp_min_(A, mini):
    logger.debug("GEMS_ENFLAME CLAMP__MIN")
    if mini is None:
        raise ValueError("Mini must not be None")
    if _flat_eligible(A, mini):
        n_total = A.numel()
        if n_total == 0:
            return A
        block, num_warps, body, cap = _choose_cfg(n_total, A.dtype, True)
        grid = (min(triton.cdiv(n_total, block), cap),)
        with torch_device_fn.device(A.device):
            _launch_flat((A, mini, A), grid, block, n_total, num_warps, body)
        return A
    if isinstance(mini, torch.Tensor):
        if mini.device != A.device:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, "
                f"but found at least two devices, {A.device} and {mini.device}"
            )
        return clamp_func_min_tensor(A, mini, out0=A)
    return clamp_func_min(A, mini, out0=A)


def clamp(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLAMP")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max(A, maxi)
    elif maxi is None:
        return clamp_func_min(A, mini)
    else:
        return clamp_func(A, mini, maxi)


def clamp_(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLAMP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    elif mini is None:
        return clamp_func_max(A, maxi, out0=A)
    elif maxi is None:
        return clamp_func_min(A, mini, out0=A)
    else:
        return clamp_func(A, mini, maxi, out0=A)
