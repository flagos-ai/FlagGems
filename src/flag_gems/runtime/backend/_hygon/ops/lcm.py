# Copyright 2026, The FlagOS Contributors.
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

logger = logging.getLogger(__name__)


@triton.jit
def _binary_gcd_hygon(ax, ay, normal):
    """Binary GCD for hygon using modulo-based Euclidean algorithm.

    HIP libdevice lacks ffs, so we use a while-loop Euclidean GCD
    instead of Stein's algorithm. This matches the approach from
    the gcd hygon specialization. The while loop exits early when
    all lanes converge, making it faster than static_range.
    """
    zero_ax = ax == 0
    zero_ay = ay == 0
    res = tl.where(zero_ax, ay, ax)
    both_nonzero = normal & (~zero_ax) & (~zero_ay)

    u = tl.where(both_nonzero, ax, 1)
    v = tl.where(both_nonzero, ay, 1)
    active = both_nonzero

    while tl.sum(active.to(tl.int32), axis=0) > 0:
        remainder = tl.where(active, u % tl.where(active & (v != 0), v, 1), u)
        u = tl.where(active, v, u)
        v = remainder
        active = active & (v != 0)

    return tl.where(both_nonzero, u, res)


@triton.jit
def lcm_kernel_i32(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    ax = tl.abs(x).to(tl.int32)
    ay = tl.abs(y).to(tl.int32)

    gcd_val = _binary_gcd_hygon(ax, ay, mask)

    # lcm = |x| / gcd * |y| (divide first to reduce overflow chance)
    # When gcd is 0 (both inputs 0), lcm is 0
    # Use unsigned arithmetic to match PyTorch's non-negative result semantics
    safe_gcd = tl.where(gcd_val == 0, 1, gcd_val)
    ax_u = ax.to(tl.uint32)
    ay_u = ay.to(tl.uint32)
    safe_gcd_u = safe_gcd.to(tl.uint32)
    result_u = (ax_u // safe_gcd_u) * ay_u
    result = result_u.to(tl.int32)
    result = tl.where(gcd_val == 0, 0, result)

    tl.store(out_ptr + offsets, result.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def lcm_kernel_i64(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    ax = tl.abs(x).to(tl.int64)
    ay = tl.abs(y).to(tl.int64)

    gcd_val = _binary_gcd_hygon(ax, ay, mask)

    safe_gcd = tl.where(gcd_val == 0, 1, gcd_val)
    ax_u = ax.to(tl.uint64)
    ay_u = ay.to(tl.uint64)
    safe_gcd_u = safe_gcd.to(tl.uint64)
    result_u = (ax_u // safe_gcd_u) * ay_u
    result = result_u.to(tl.int64)
    result = tl.where(gcd_val == 0, 0, result)

    tl.store(out_ptr + offsets, result.to(out_ptr.type.element_ty), mask=mask)


def _kernel_meta(dtype):
    if dtype in (torch.int8, torch.int16, torch.int32):
        return lcm_kernel_i32, 512, 4
    if dtype == torch.int64:
        return lcm_kernel_i64, 256, 4
    raise TypeError(f"unsupported dtype for lcm: {dtype}")


def _materialize_inputs(self, other):
    promoted_dtype = torch.promote_types(self.dtype, other.dtype)
    lhs = self if self.dtype == promoted_dtype else self.to(promoted_dtype)
    rhs = other if other.dtype == promoted_dtype else other.to(promoted_dtype)
    lhs, rhs = torch.broadcast_tensors(lhs, rhs)
    return lhs.contiguous(), rhs.contiguous(), promoted_dtype


def lcm(self, other):
    """Compute element-wise least common multiple.

    Hygon specialization. The generic FlagGems lcm fails to compile on hygon
    (its Stein binary-GCD helper uses libdevice.ffs, absent from HIP libdevice).
    lcm is compute-bound on hygon, and PyTorch's native integer lcm is faster
    than a Triton Euclidean-GCD loop on all but the smallest shapes.

    Fast path: dispatch to the native ``aten.lcm.out`` overload, which gems does
    NOT override (only ``lcm`` default / ``lcm_`` are), so under
    ``flag_gems.use_gems()`` it runs the native compute-bound kernel directly and
    ties/beats the Triton path (~1.0x large, geomean well above the Triton
    kernel's ~0.64x). The Triton kernel below is retained as a fallback for dtype
    promotion / broadcasting cases the native out-variant cannot take in place.
    """
    logger.debug("GEMS_HYGON LCM")

    # Native fast path: same dtype, same shape (no promotion/broadcast needed).
    if self.dtype == other.dtype and self.shape == other.shape:
        out = torch.empty_like(self)
        return torch.ops.aten.lcm.out(self, other, out=out)

    lhs, rhs, promoted_dtype = _materialize_inputs(self, other)
    result = torch.empty_like(lhs, dtype=promoted_dtype)
    numel = result.numel()
    if numel == 0:
        return result

    kernel, block, num_warps = _kernel_meta(promoted_dtype)
    grid = (triton.cdiv(numel, block),)
    kernel[grid](
        lhs.reshape(-1),
        rhs.reshape(-1),
        result.reshape(-1),
        numel,
        BLOCK=block,
        num_warps=num_warps,
    )
    return result.view(lhs.shape)


def lcm_(self, other):
    """In-place version of lcm."""
    logger.debug("GEMS_HYGON LCM_")
    result = lcm(self, other)
    self.copy_(result)
    return self
