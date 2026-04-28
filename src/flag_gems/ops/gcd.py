<<<<<<< feature/FlagGemsOperatorDevelopmentCompetition
"""
GCD operator implementation for FlagGems.

This module provides the greatest common divisor (GCD) operation using native Triton kernels.
Implements optimized Euclidean algorithm with Lamé's theorem for iteration estimation.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def gcd_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data with cache hint
    a = tl.load(a_ptr + offsets, mask=mask, other=0, cache_modifier=".cg")
    b = tl.load(b_ptr + offsets, mask=mask, other=0, cache_modifier=".cg")

    # GCD is always computed on absolute values
    a = tl.abs(a)
    b = tl.abs(b)

    # Find maximum value in block to estimate required iterations
    # Based on Lamé's theorem: iterations ≤ 5 × number of digits
    min_ab = tl.where(a < b, a, b)
    max_val = tl.max(min_ab)

    # Estimate iterations based on magnitude ranges
    if max_val < 10:
        max_iter = 5
    elif max_val < 100:
        max_iter = 10
    elif max_val < 1000:
        max_iter = 15
    elif max_val < 10000:
        max_iter = 20
    elif max_val < 100000:
        max_iter = 25
    elif max_val < 1000000:
        max_iter = 30
    elif max_val < 10000000:
        max_iter = 35
    else:
        max_iter = 45

    # Standard Euclidean algorithm with loop unrolling (5 iterations per group)
    num_unrolled_groups = max_iter // 5
    remaining_iters = max_iter % 5

    # Unrolled loop groups
    for _ in range(num_unrolled_groups):
        # Iteration 1
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

        # Iteration 2
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

        # Iteration 3
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

        # Iteration 4
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

        # Iteration 5
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

    # Remaining iterations
    for _ in range(remaining_iters):
        converged = b == 0
        b_safe = tl.where(converged, 1, b)
        remainder = a % b_safe
        a = tl.where(converged, a, b)
        b = tl.where(converged, b, remainder)

    # Result is in 'a'
    result = a

    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def gcd(input, other):
    """
    Compute the greatest common divisor (GCD) of two tensors element-wise.

    This function provides the GCD operation for FlagGems, optimized
    for GPU execution using native Triton kernels with loop unrolling
    and Lamé's theorem for iteration estimation.

    Args:
        input (Tensor): First input tensor
        other (Tensor): Second input tensor

    Returns:
        Tensor: Tensor containing GCD values

    Examples:
        >>> import torch
        >>> from flag_gems import gcd
        >>> x = torch.tensor([12, 18, 24], device='cuda')
        >>> y = torch.tensor([8, 12, 36], device='cuda')
        >>> z = gcd(x, y)
        >>> print(z)
        tensor([4, 6, 12], device='cuda')
    """
    logger.debug("GEMS GCD")

    # Handle empty tensor
    if input.numel() == 0:
        return torch.empty_like(input)

    # Ensure inputs have the same shape
    if input.shape != other.shape:
        raise ValueError(
            f"Shape mismatch: input.shape={input.shape}, other.shape={other.shape}"
        )

    # Create output tensor
    out = torch.empty_like(input)

    # Flatten tensors for 1D processing
    input_flat = input.contiguous().view(-1)
    other_flat = other.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)

    # Get total number of elements
    n_elements = input_flat.numel()

    # Choose optimal block size based on input size
    if n_elements < 1024:
        BLOCK_SIZE = 256
    elif n_elements < 16384:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024

    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_size,)

    # Launch kernel
    gcd_kernel[grid](
        input_flat,
        other_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
=======
import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

logger = logging.getLogger(__name__)
_I16_MIN_LUT_CACHE = {}


@triton.jit
def _ctz(x):
    return libdevice.ffs(x) - 1


@triton.jit
def _abs_u32(x):
    ux = x.to(tl.uint32)
    return tl.where(x < 0, 0 - ux, ux)


@triton.jit
def _abs_u64(x):
    ux = x.to(tl.uint64)
    return tl.where(x < 0, 0 - ux, ux)


@triton.jit
def _c_rem_i32(a, b):
    mag = _abs_u32(a) % _abs_u32(b)
    rem = mag.to(tl.int32)
    return tl.where((a < 0) & (mag != 0), -rem, rem)


@triton.jit
def _c_rem_i64(a, b):
    mag = _abs_u64(a) % _abs_u64(b)
    rem = mag.to(tl.int64)
    return tl.where((a < 0) & (mag != 0), -rem, rem)


@triton.jit
def _binary_gcd(ax, ay, normal):
    zero_ax = ax == 0
    zero_ay = ay == 0
    res = tl.where(zero_ax, ay, ax)
    both_nonzero = normal & (~zero_ax) & (~zero_ay)
    common = _ctz(tl.where(both_nonzero, ax | ay, 1))
    u = tl.where(both_nonzero, ax >> _ctz(tl.where(both_nonzero, ax, 1)), ax)
    v = ay
    active = both_nonzero

    while tl.sum(active.to(tl.int32), axis=0) > 0:
        v_shifted = tl.where(active, v >> _ctz(tl.where(active, v, 1)), v)
        swap = active & (u > v_shifted)
        small = tl.where(swap, v_shifted, u)
        large = tl.where(swap, u, v_shifted)
        u = tl.where(active, small, u)
        v = tl.where(active, large - small, v)
        active = active & (v != 0)

    return tl.where(both_nonzero, u << common, res)


@triton.jit
def gcd_kernel_i16(x_ptr, y_ptr, lut_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    x_i32 = x.to(tl.int32)
    y_i32 = y.to(tl.int32)
    min_value: tl.constexpr = -32768
    min_x = x_i32 == min_value
    min_y = y_i32 == min_value
    special_mask = mask & (min_x | min_y)
    normal = mask & (~special_mask)
    ax = tl.abs(x_i32)
    ay = tl.abs(y_i32)
    normal_res = _binary_gcd(ax, ay, normal)

    both_min = special_mask & min_x & min_y
    one_min = special_mask & (~both_min)
    other_abs = tl.where(min_x, tl.abs(y_i32), tl.abs(x_i32))
    special_res = tl.load(lut_ptr + other_abs, mask=one_min, other=0).to(tl.int32)
    special_res = tl.where(both_min, min_value, special_res)

    out = tl.where(special_mask, special_res, normal_res)
    tl.store(out_ptr + offsets, out.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def gcd_kernel_i32(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    min_value: tl.constexpr = -(1 << 31)
    min_x = x == min_value
    min_y = y == min_value
    ax_native = tl.where(min_x, x, tl.abs(x))
    ay_native = tl.where(min_y, y, tl.abs(y))
    ax = ax_native.to(tl.int32)
    ay = ay_native.to(tl.int32)

    special_mask = mask & (min_x | min_y)
    normal = mask & (~special_mask)
    normal_res = _binary_gcd(ax, ay, normal)

    sa = ax_native.to(tl.int32)
    sb = ay_native.to(tl.int32)
    special = special_mask & (sa != 0)
    while tl.sum(special.to(tl.int32), axis=0) > 0:
        next_sa = tl.where(special, _c_rem_i32(sb, tl.where(special, sa, 1)), sa)
        sb = tl.where(special, sa, sb)
        sa = next_sa
        special = special & (sa != 0)

    out = tl.where(mask & (~normal), sb, normal_res)
    tl.store(out_ptr + offsets, out.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def gcd_kernel_i64(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    min_x = x == -(1 << 63)
    min_y = y == -(1 << 63)
    ax = tl.where(min_x, x, tl.abs(x)).to(tl.int64)
    ay = tl.where(min_y, y, tl.abs(y)).to(tl.int64)

    special_mask = mask & (min_x | min_y)
    normal = mask & (~special_mask)
    normal_res = _binary_gcd(ax, ay, normal)

    sa = ax
    sb = ay
    special = special_mask & (sa != 0)
    while tl.sum(special.to(tl.int32), axis=0) > 0:
        next_sa = tl.where(special, _c_rem_i64(sb, tl.where(special, sa, 1)), sa)
        sb = tl.where(special, sa, sb)
        sa = next_sa
        special = special & (sa != 0)

    out = tl.where(mask & (~normal), sb, normal_res)
    tl.store(out_ptr + offsets, out.to(out_ptr.type.element_ty), mask=mask)


def _kernel_meta(dtype):
    if dtype == torch.int16:
        return gcd_kernel_i16, 512, 4
    if dtype == torch.int32:
        return gcd_kernel_i32, 512, 4
    if dtype == torch.int64:
        return gcd_kernel_i64, 256, 4
    raise TypeError(f"unsupported dtype for gcd: {dtype}")


def _get_i16_min_lut(device):
    key = (device.type, device.index)
    lut = _I16_MIN_LUT_CACHE.get(key)
    if lut is None:
        info = torch.iinfo(torch.int16)
        lhs = torch.full((info.max + 1,), info.min, dtype=torch.int16)
        rhs = torch.arange(info.max + 1, dtype=torch.int16)
        lut = torch.gcd(lhs, rhs).to(device=device)
        _I16_MIN_LUT_CACHE[key] = lut
    return lut


def _materialize_inputs(self, other):
    promoted_dtype = torch.promote_types(self.dtype, other.dtype)
    lhs = self if self.dtype == promoted_dtype else self.to(promoted_dtype)
    rhs = other if other.dtype == promoted_dtype else other.to(promoted_dtype)
    lhs, rhs = torch.broadcast_tensors(lhs, rhs)
    return lhs.contiguous(), rhs.contiguous(), promoted_dtype


def _launch_gcd(lhs, rhs, out):
    numel = out.numel()
    if numel == 0:
        return out

    kernel, block, num_warps = _kernel_meta(out.dtype)
    grid = (triton.cdiv(numel, block),)
    if out.dtype == torch.int16:
        lut = _get_i16_min_lut(out.device)
        kernel[grid](lhs, rhs, lut, out, numel, BLOCK=block, num_warps=num_warps)
    else:
        kernel[grid](lhs, rhs, out, numel, BLOCK=block, num_warps=num_warps)
    return out


def gcd(self, other, *, out=None):
    logger.debug("GEMS GCD")
    lhs, rhs, promoted_dtype = _materialize_inputs(self, other)
    result = torch.empty_like(lhs, dtype=promoted_dtype)
    _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), result.reshape(-1))
    result = result.view(lhs.shape)
    if out is None:
        return result

    out.copy_(result)
    return out


def gcd_out(self, other, *, out=None):
    logger.debug("GEMS GCD_OUT")
    if out is None:
        return gcd(self, other)
    return gcd(self, other, out=out)
>>>>>>> master
