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
import numbers
import struct

import torch
import triton
import triton.language as tl
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.codegen_config_utils import get_codegen_config
from flag_gems.utils.shape_utils import (
    MemOverlap,
    broadcast_shapes,
    broadcasted_stride,
    has_internal_overlapping,
)
from flag_gems.utils.type_utils import type_promotion

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
}
_CODEGEN_CONFIG = get_codegen_config()
_MAX_BLOCK_SIZE = _CODEGEN_CONFIG.max_tile_size
_MAX_GRID_SIZE = _CODEGEN_CONFIG.max_grid_size[0]


@triton.jit
def _laguerre_polynomial_l(x, n, valid, use_fp64: tl.constexpr):
    compute_dtype = tl.float64 if use_fp64 else tl.float32
    lane_zeros = tl.zeros(valid.shape, dtype=compute_dtype)
    x = x.to(compute_dtype) + lane_zeros
    n_float = n.to(compute_dtype) + lane_zeros

    # C++ leaves non-finite and out-of-range float-to-int conversions undefined.
    # Map them to a negative degree, matching the observable CPU result while
    # preventing an invalid conversion from becoming a watchdog-length loop.
    n_is_nan = n_float != n_float
    n_is_pos_inf = n_float == float("inf")
    n_is_neg_inf = n_float == float("-inf")
    n_is_out_of_range = (n_float >= 9223372036854775808.0) | (
        n_float < -9223372036854775808.0
    )
    n_is_invalid = n_is_nan | n_is_pos_inf | n_is_neg_inf | n_is_out_of_range
    n_int = tl.where(n_is_invalid, -1.0, n_float).to(tl.int64)

    one = tl.full(x.shape, 1.0, dtype=compute_dtype)
    zero = tl.full(x.shape, 0.0, dtype=compute_dtype)

    # q is initialized on every path: x=NaN,n>=2 cannot observe an
    # uninitialized local as it could in the old upstream control flow.
    p = one
    q = one - x

    # Common n=2/3/4 fast paths. Keep recurrence ordering for +/-inf inputs.
    tile_degree = tl.max(
        tl.where(valid & (n_int > 0) & (x != 0.0) & (q == q), n_int, 0)
    )
    if tile_degree > 1:
        active = (n_int > 1) & (q == q)
        candidate = ((3.0 - x) * q - p) / 2.0
        next_p = tl.where(active, q, p)
        q = tl.where(active, candidate, q)
        p = next_p

    if tile_degree > 2:
        active = (n_int > 2) & (q == q)
        candidate = ((5.0 - x) * q - 2.0 * p) / 3.0
        next_p = tl.where(active, q, p)
        q = tl.where(active, candidate, q)
        p = next_p

    if tile_degree > 3:
        active = (n_int > 3) & (q == q)
        candidate = ((7.0 - x) * q - 3.0 * p) / 4.0
        next_p = tl.where(active, q, p)
        q = tl.where(active, candidate, q)
        p = next_p

    # Masked tail lanes and complete special values contribute only 4 to the
    # tile bound. Thus x==0, negative n and NaN/Inf stop immediately.
    needs_loop = valid & (n_int > 4) & (x != 0.0) & (q == q)
    loop_limit = tl.max(tl.where(needs_loop, n_int, 4))
    for k in range(4, loop_limit):
        active = (n_int > k) & (q == q)
        candidate = ((2.0 * k + 1.0 - x) * q - k * p) / (k + 1.0)
        next_p = tl.where(active, q, p)
        q = tl.where(active, candidate, q)
        p = next_p

    result = q
    result = tl.where((n_int == 0) | ((x == 0.0) & (n_int >= 0)), one, result)
    result = tl.where(n_int < 0, zero, result)

    return result


@libentry()
@triton.jit
def _laguerre_flat_tensor_tensor_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    tiles_per_program,
    X_SCALAR: tl.constexpr,
    N_SCALAR: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    for tile in range(0, tiles_per_program):
        offsets = (pid + tile * num_programs) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x_offsets = tl.zeros_like(offsets) if X_SCALAR else offsets
        n_offsets = tl.zeros_like(offsets) if N_SCALAR else offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        n = tl.load(n_ptr + n_offsets, mask=mask, other=0.0)
        result = _laguerre_polynomial_l(x, n, mask, USE_FP64)
        tl.store(out_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def _laguerre_generic_tensor_tensor_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    numel,
    tiles_per_program,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    N_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    for tile in range(0, tiles_per_program):
        offsets = (pid + tile * num_programs) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        linear = offsets.to(tl.int64)
        x_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        n_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        for dim in tl.static_range(NDIM - 1, -1, -1):
            quotient = linear // SHAPE[dim]
            index = linear - quotient * SHAPE[dim]
            linear = quotient
            x_offsets += index * X_STRIDES[dim]
            n_offsets += index * N_STRIDES[dim]
            out_offsets += index * OUT_STRIDES[dim]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        n = tl.load(n_ptr + n_offsets, mask=mask, other=0.0)
        result = _laguerre_polynomial_l(x, n, mask, USE_FP64)
        tl.store(out_ptr + out_offsets, result, mask=mask)


@libentry()
@triton.jit
def _laguerre_generic_tensor_tensor_runtime_meta_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    meta_ptr,
    numel,
    tiles_per_program,
    NDIM: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    for tile in range(0, tiles_per_program):
        offsets = (pid + tile * num_programs) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        linear = offsets.to(tl.int64)
        x_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        n_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        for dim in tl.static_range(NDIM - 1, -1, -1):
            shape_dim = tl.load(meta_ptr + dim)
            quotient = linear // shape_dim
            index = linear - quotient * shape_dim
            linear = quotient
            x_offsets += index * tl.load(meta_ptr + NDIM + dim)
            n_offsets += index * tl.load(meta_ptr + 2 * NDIM + dim)
            out_offsets += index * tl.load(meta_ptr + 3 * NDIM + dim)
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        n = tl.load(n_ptr + n_offsets, mask=mask, other=0.0)
        result = _laguerre_polynomial_l(x, n, mask, USE_FP64)
        tl.store(out_ptr + out_offsets, result, mask=mask)


@libentry()
@triton.jit
def _laguerre_flat_tensor_scalar_kernel(
    tensor_ptr,
    scalar,
    out_ptr,
    numel,
    tiles_per_program,
    SCALAR_IS_X: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    if USE_FP64:
        scalar = scalar.to(tl.int64).to(tl.float64, bitcast=True)
    for tile in range(0, tiles_per_program):
        offsets = (pid + tile * num_programs) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        tensor = tl.load(tensor_ptr + offsets, mask=mask, other=0.0)
        x = scalar if SCALAR_IS_X else tensor
        n = tensor if SCALAR_IS_X else scalar
        result = _laguerre_polynomial_l(x, n, mask, USE_FP64)
        tl.store(out_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit
def _laguerre_generic_tensor_scalar_kernel(
    tensor_ptr,
    scalar,
    out_ptr,
    numel,
    tiles_per_program,
    SHAPE: tl.constexpr,
    TENSOR_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    SCALAR_IS_X: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    if USE_FP64:
        scalar = scalar.to(tl.int64).to(tl.float64, bitcast=True)
    for tile in range(0, tiles_per_program):
        offsets = (pid + tile * num_programs) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        linear = offsets.to(tl.int64)
        tensor_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        for dim in tl.static_range(NDIM - 1, -1, -1):
            quotient = linear // SHAPE[dim]
            index = linear - quotient * SHAPE[dim]
            linear = quotient
            tensor_offsets += index * TENSOR_STRIDES[dim]
            out_offsets += index * OUT_STRIDES[dim]
        tensor = tl.load(tensor_ptr + tensor_offsets, mask=mask, other=0.0)
        x = scalar if SCALAR_IS_X else tensor
        n = tensor if SCALAR_IS_X else scalar
        result = _laguerre_polynomial_l(x, n, mask, USE_FP64)
        tl.store(out_ptr + out_offsets, result, mask=mask)


@libentry()
@triton.jit
def _laguerre_generic_tensor_scalar_runtime_meta_kernel(
    tensor_ptr,
    scalar,
    out_ptr,
    meta_ptr,
    numel,
    tiles_per_program,
    NDIM: tl.constexpr,
    SCALAR_IS_X: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    num_programs = tle.num_programs(0)
    if USE_FP64:
        scalar = scalar.to(tl.int64).to(tl.float64, bitcast=True)
    for tile in range(0, tiles_per_program):
        offsets = (pid + tile * num_programs) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        linear = offsets.to(tl.int64)
        tensor_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
        for dim in tl.static_range(NDIM - 1, -1, -1):
            shape_dim = tl.load(meta_ptr + dim)
            quotient = linear // shape_dim
            index = linear - quotient * shape_dim
            linear = quotient
            tensor_offsets += index * tl.load(meta_ptr + NDIM + dim)
            out_offsets += index * tl.load(meta_ptr + 2 * NDIM + dim)
        tensor = tl.load(tensor_ptr + tensor_offsets, mask=mask, other=0.0)
        x = scalar if SCALAR_IS_X else tensor
        n = tensor if SCALAR_IS_X else scalar
        result = _laguerre_polynomial_l(x, n, mask, USE_FP64)
        tl.store(out_ptr + out_offsets, result, mask=mask)


def _validate_and_get_result_dtype(x, n):
    tensors = []
    for value in (x, n):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
            if value.dtype not in _SUPPORTED_DTYPES:
                raise RuntimeError(
                    "special_laguerre_polynomial_l only supports float32, "
                    f"float64, integral and bool inputs, got {value.dtype}"
                )
        elif not isinstance(value, numbers.Real):
            raise TypeError(
                "special_laguerre_polynomial_l expects real scalar arguments, "
                f"got {type(value).__name__}"
            )

    if not tensors:
        raise TypeError("special_laguerre_polynomial_l has no scalar/scalar overload")
    if any(tensor.device != tensors[0].device for tensor in tensors[1:]):
        raise RuntimeError("special_laguerre_polynomial_l inputs must share a device")

    _, result_dtype = type_promotion(
        x,
        n,
        type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    if result_dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "special_laguerre_polynomial_l only supports float32 and float64 "
            f"computation, got {result_dtype}"
        )
    return tensors[0], result_dtype


def _prepare_output(x, n, tensor, result_dtype, out):
    shapes = [value.shape for value in (x, n) if isinstance(value, torch.Tensor)]
    output_shape = broadcast_shapes(shapes)
    if out is None:
        out = torch.empty(output_shape, dtype=result_dtype, device=tensor.device)
    else:
        if out.device != tensor.device:
            raise RuntimeError(
                "special_laguerre_polynomial_l out must share the input device"
            )
        if tuple(out.shape) != tuple(output_shape):
            out.resize_(output_shape)
        if has_internal_overlapping(out) == MemOverlap.Yes:
            raise RuntimeError("special_laguerre_polynomial_l out has internal overlap")
        if not torch.can_cast(result_dtype, out.dtype):
            raise RuntimeError(
                f"result type {result_dtype} can't be cast to {out.dtype}"
            )
    return out, output_shape


def _launch_geometry(numel):
    block_size = min(_MAX_BLOCK_SIZE, triton.next_power_of_2(max(1, numel)))
    num_tiles = triton.cdiv(numel, block_size)
    num_programs = min(_MAX_GRID_SIZE, num_tiles)
    tiles_per_program = triton.cdiv(num_tiles, num_programs)
    return (num_programs,), block_size, tiles_per_program


def _triton_version_lt(major, minor):
    version = triton.__version__.split("+", 1)[0]
    parts = version.split(".")
    try:
        current = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):
        return False
    return current < (major, minor)


def _needs_runtime_meta_for_constexpr_tuple():
    if _triton_version_lt(3, 3):
        return True
    try:
        import triton.language.core as tl_core

        frontend_tuple = getattr(tl_core, "tuple", None)
        return frontend_tuple is None or not hasattr(frontend_tuple, "__getitem__")
    except Exception:
        return False


def _scalar_kernel_arg(scalar, result_dtype):
    if result_dtype != torch.float64:
        return scalar
    return struct.unpack("=q", struct.pack("=d", float(scalar)))[0]


def _dispatch(x, n, out=None):
    tensor, result_dtype = _validate_and_get_result_dtype(x, n)
    out, output_shape = _prepare_output(x, n, tensor, result_dtype, out)
    if out.numel() == 0:
        return out

    grid, block_size, tiles_per_program = _launch_geometry(out.numel())
    use_fp64 = result_dtype == torch.float64
    with torch_device_fn.device(tensor.device.index):
        if isinstance(x, torch.Tensor) and isinstance(n, torch.Tensor):
            x_scalar = x.numel() == 1
            n_scalar = n.numel() == 1
            flat = (
                out.is_contiguous()
                and (x_scalar or (tuple(x.shape) == output_shape and x.is_contiguous()))
                and (n_scalar or (tuple(n.shape) == output_shape and n.is_contiguous()))
            )
            if flat:
                _laguerre_flat_tensor_tensor_kernel[grid](
                    x,
                    n,
                    out,
                    out.numel(),
                    tiles_per_program,
                    X_SCALAR=x_scalar,
                    N_SCALAR=n_scalar,
                    USE_FP64=use_fp64,
                    BLOCK_SIZE=block_size,
                )
            else:
                shape = tuple(output_shape)
                x_strides = broadcasted_stride(x.shape, x.stride(), output_shape)
                n_strides = broadcasted_stride(n.shape, n.stride(), output_shape)
                out_strides = tuple(out.stride())
                if _needs_runtime_meta_for_constexpr_tuple():
                    meta = torch.tensor(
                        shape + x_strides + n_strides + out_strides,
                        dtype=torch.int64,
                        device=tensor.device,
                    )
                    _laguerre_generic_tensor_tensor_runtime_meta_kernel[grid](
                        x,
                        n,
                        out,
                        meta,
                        out.numel(),
                        tiles_per_program,
                        NDIM=len(output_shape),
                        USE_FP64=use_fp64,
                        BLOCK_SIZE=block_size,
                    )
                else:
                    _laguerre_generic_tensor_tensor_kernel[grid](
                        x,
                        n,
                        out,
                        out.numel(),
                        tiles_per_program,
                        SHAPE=shape,
                        X_STRIDES=x_strides,
                        N_STRIDES=n_strides,
                        OUT_STRIDES=out_strides,
                        NDIM=len(output_shape),
                        USE_FP64=use_fp64,
                        BLOCK_SIZE=block_size,
                    )
        else:
            scalar_is_x = not isinstance(x, torch.Tensor)
            input_tensor = n if scalar_is_x else x
            scalar = x if scalar_is_x else n
            scalar = _scalar_kernel_arg(scalar, result_dtype)
            flat = (
                input_tensor.is_contiguous()
                and out.is_contiguous()
                and tuple(input_tensor.shape) == output_shape
            )
            if flat:
                _laguerre_flat_tensor_scalar_kernel[grid](
                    input_tensor,
                    scalar,
                    out,
                    out.numel(),
                    tiles_per_program,
                    SCALAR_IS_X=scalar_is_x,
                    USE_FP64=use_fp64,
                    BLOCK_SIZE=block_size,
                )
            else:
                shape = tuple(output_shape)
                tensor_strides = broadcasted_stride(
                    input_tensor.shape, input_tensor.stride(), output_shape
                )
                out_strides = tuple(out.stride())
                if _needs_runtime_meta_for_constexpr_tuple():
                    meta = torch.tensor(
                        shape + tensor_strides + out_strides,
                        dtype=torch.int64,
                        device=tensor.device,
                    )
                    _laguerre_generic_tensor_scalar_runtime_meta_kernel[grid](
                        input_tensor,
                        scalar,
                        out,
                        meta,
                        out.numel(),
                        tiles_per_program,
                        NDIM=len(output_shape),
                        SCALAR_IS_X=scalar_is_x,
                        USE_FP64=use_fp64,
                        BLOCK_SIZE=block_size,
                    )
                else:
                    _laguerre_generic_tensor_scalar_kernel[grid](
                        input_tensor,
                        scalar,
                        out,
                        out.numel(),
                        tiles_per_program,
                        SHAPE=shape,
                        TENSOR_STRIDES=tensor_strides,
                        OUT_STRIDES=out_strides,
                        NDIM=len(output_shape),
                        SCALAR_IS_X=scalar_is_x,
                        USE_FP64=use_fp64,
                        BLOCK_SIZE=block_size,
                    )
    return out


def special_laguerre_polynomial_l(x, n):
    logger.debug("GEMS SPECIAL_LAGUERRE_POLYNOMIAL_L")
    return _dispatch(x, n)


def special_laguerre_polynomial_l_out(x, n, out):
    logger.debug("GEMS SPECIAL_LAGUERRE_POLYNOMIAL_L_OUT")
    return _dispatch(x, n, out=out)
