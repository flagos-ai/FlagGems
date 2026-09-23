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

from flag_gems import runtime
from flag_gems.utils import tl_extra_shim
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion

logger = logging.getLogger(__name__)

_pow = tl_extra_shim.pow
_SUPPORTED_COMPUTE_DTYPES = (torch.float32, torch.float64)
_FLAT_BLOCK = 256
_MAX_FLAT_GRID = 65535


@triton.jit
def _zeta_compute(x, q):
    """Cephes Euler--Maclaurin approximation used by ATen special_zeta."""
    # Scalar overloads pass one rank-0 kernel argument.  Materialize the common
    # lane shape without arithmetic so Inf/NaN inputs keep their bit semantics.
    x, q = tl.broadcast(x, q)

    # This is deliberately 2**-53 for both compute dtypes.  ATen's CUDA float
    # kernel also converts this constant to float instead of using float eps.
    machep = 1.11022302462515654042e-16

    total = _pow(q, -x)
    a = q
    b = total
    direct_done = x < x  # all-false vector, including NaN lanes

    # Cephes always takes at least nine recurrence steps.
    for _ in range(9):
        active = ~direct_done
        next_a = a + 1.0
        next_b = _pow(next_a, -x)
        next_total = total + next_b
        converged = (-machep * next_total < next_b) & (next_b < machep * next_total)
        a = tl.where(active, next_a, a)
        b = tl.where(active, next_b, b)
        total = tl.where(active, next_total, total)
        direct_done = direct_done | (active & converged)

    # Cephes keeps recurring until a > 9.  A block-wide ballot preserves that
    # exact threshold without paying for dormant pow calls on the common q > 0
    # path; lanes that met the precision threshold above remain inactive.
    active = (~direct_done) & (a <= 9.0)
    while tl.sum(tl.ravel(active.to(tl.int32)), axis=0) > 0:
        next_a = a + 1.0
        next_b = _pow(next_a, -x)
        next_total = total + next_b
        converged = (-machep * next_total < next_b) & (next_b < machep * next_total)
        a = tl.where(active, next_a, a)
        b = tl.where(active, next_b, b)
        total = tl.where(active, next_total, total)
        direct_done = direct_done | (active & converged)
        active = (~direct_done) & (a <= 9.0)

    direct_result = total

    # Euler--Maclaurin tail followed by the twelve Cephes Bernoulli terms.
    w = a
    total = total + b * w / (x - 1.0) - 0.5 * b
    product = 1.0 + 0.0 * x
    em_done = x < x
    k = 0

    product = product * (x + k)
    b = b / w
    term = product * b / 12.0
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / -720.0
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / 30240.0
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / -1209600.0
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / 47900160.0
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / -1.8924375803183791606e9
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / 7.47242496e10
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / -2.950130727918164224e12
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / 1.1646782814350067249e14
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / -4.5979787224074726105e15
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / 1.8152105401943546773e17
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)
    em_done = em_done | (active & (tl.abs(term / candidate) < machep))
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1

    product = product * (x + k)
    b = b / w
    term = product * b / -7.1661652561756670113e18
    candidate = total + term
    active = ~em_done
    total = tl.where(active, candidate, total)

    result = tl.where(direct_done, direct_result, total)

    # Apply ATen's domain checks in reverse order so x == 1 has precedence.
    q_nonpositive = q <= 0.0
    q_integer = q == tl.floor(q)
    x_integer = x == tl.floor(x)
    result = tl.where(q_nonpositive & (~q_integer) & (~x_integer), float("nan"), result)
    result = tl.where(q_nonpositive & q_integer, float("inf"), result)
    result = tl.where(x < 1.0, float("nan"), result)
    # Some Ascend compiler versions lose the direct-series early-return mask
    # after NaNs are formed in the dormant Euler tail.  This is the one finite
    # x=+inf result produced by Cephes: pow(1, -inf) == 1.
    result = tl.where((x == float("inf")) & (q == 1.0), 1.0, result)
    return tl.where(x == 1.0, float("inf"), result)


@triton.jit
def special_zeta_flat_kernel(
    x,
    q,
    out,
    n_elements,
    X_IS_TENSOR: tl.constexpr,
    Q_IS_TENSOR: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK
    grid_stride = tl.num_programs(0) * BLOCK
    while block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        if X_IS_TENSOR:
            x_value = tl.load(x + offsets, mask=mask, other=1.0)
        else:
            x_value = x
        if Q_IS_TENSOR:
            q_value = tl.load(q + offsets, mask=mask, other=1.0)
        else:
            q_value = q
        if FP64:
            compute_dtype = tl.float64
        else:
            compute_dtype = tl.float32
        result = _zeta_compute(x_value.to(compute_dtype), q_value.to(compute_dtype))
        tl.store(out + offsets, result, mask=mask)
        block_start += grid_stride


@triton.jit
def special_zeta_strided_kernel(
    x,
    q,
    out,
    meta,
    n_elements,
    NDIM: tl.constexpr,
    X_IS_TENSOR: tl.constexpr,
    Q_IS_TENSOR: tl.constexpr,
    FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK
    grid_stride = tl.num_programs(0) * BLOCK
    while block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        linear = offsets
        x_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        q_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        out_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        for dim in tl.static_range(NDIM - 1, -1, -1):
            shape_dim = tl.load(meta + dim)
            index = linear % shape_dim
            linear = linear // shape_dim
            x_offsets += index * tl.load(meta + NDIM + dim)
            q_offsets += index * tl.load(meta + 2 * NDIM + dim)
            out_offsets += index * tl.load(meta + 3 * NDIM + dim)
        if X_IS_TENSOR:
            x_value = tl.load(x + x_offsets, mask=mask, other=1.0)
        else:
            x_value = x
        if Q_IS_TENSOR:
            q_value = tl.load(q + q_offsets, mask=mask, other=1.0)
        else:
            q_value = q
        if FP64:
            compute_dtype = tl.float64
        else:
            compute_dtype = tl.float32
        result = _zeta_compute(x_value.to(compute_dtype), q_value.to(compute_dtype))
        tl.store(out + out_offsets, result, mask=mask)
        block_start += grid_stride


def _launch_flat(x, q, out, dtype, *, x_is_tensor, q_is_tensor):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = (min(triton.cdiv(n_elements, _FLAT_BLOCK), _MAX_FLAT_GRID),)
    with runtime.torch_device_fn.device(out.device):
        special_zeta_flat_kernel[grid](
            x,
            q,
            out,
            n_elements,
            X_IS_TENSOR=x_is_tensor,
            Q_IS_TENSOR=q_is_tensor,
            FP64=dtype == torch.float64,
            BLOCK=_FLAT_BLOCK,
            num_warps=4,
        )
    return out


def _broadcasted_stride(tensor, out_shape):
    offset = len(out_shape) - tensor.ndim
    result = []
    for out_dim, out_size in enumerate(out_shape):
        in_dim = out_dim - offset
        if in_dim < 0:
            result.append(0)
            continue
        size = tensor.shape[in_dim]
        result.append(0 if size == 1 and out_size != 1 else tensor.stride(in_dim))
    return tuple(result)


def _launch_strided(x, q, out, dtype, *, x_is_tensor, q_is_tensor):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    out_shape = tuple(out.shape) or (1,)
    ndim = len(out_shape)
    x_stride = (
        _broadcasted_stride(x, tuple(out.shape)) or (0,) if x_is_tensor else (0,) * ndim
    )
    q_stride = (
        _broadcasted_stride(q, tuple(out.shape)) or (0,) if q_is_tensor else (0,) * ndim
    )
    out_stride = tuple(out.stride()) or (0,)
    meta = torch.tensor(
        out_shape + x_stride + q_stride + out_stride,
        device=out.device,
        dtype=torch.int32,
    )
    grid = (min(triton.cdiv(n_elements, _FLAT_BLOCK), _MAX_FLAT_GRID),)
    with runtime.torch_device_fn.device(out.device):
        special_zeta_strided_kernel[grid](
            x,
            q,
            out,
            meta,
            n_elements,
            NDIM=ndim,
            X_IS_TENSOR=x_is_tensor,
            Q_IS_TENSOR=q_is_tensor,
            FP64=dtype == torch.float64,
            BLOCK=_FLAT_BLOCK,
            num_warps=4,
        )
    return out


def _launch_layout(x, q, out, dtype, *, x_is_tensor, q_is_tensor):
    tensor_inputs = [value for value in (x, q) if isinstance(value, torch.Tensor)]
    if (
        all(
            tensor.shape == out.shape and tensor.is_contiguous()
            for tensor in tensor_inputs
        )
        and out.is_contiguous()
    ):
        return _launch_flat(
            x,
            q,
            out,
            dtype,
            x_is_tensor=x_is_tensor,
            q_is_tensor=q_is_tensor,
        )
    return _launch_strided(
        x,
        q,
        out,
        dtype,
        x_is_tensor=x_is_tensor,
        q_is_tensor=q_is_tensor,
    )


def _compute_result(x, q, shape, device, dtype, *, x_is_tensor, q_is_tensor):
    out = torch.empty(tuple(shape), device=device, dtype=dtype)
    return _launch_layout(
        x,
        q,
        out,
        dtype,
        x_is_tensor=x_is_tensor,
        q_is_tensor=q_is_tensor,
    )


def _promoted_dtype(x, q):
    _, result_dtype = type_promotion(
        x,
        q,
        type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    if result_dtype not in _SUPPORTED_COMPUTE_DTYPES:
        raise RuntimeError(
            "special_zeta kernel only supports a promoted float32 or float64 "
            f"dtype, but got {result_dtype}"
        )
    if result_dtype == torch.float64 and not runtime.device.support_fp64:
        raise RuntimeError(
            f"special_zeta does not support float64 on {runtime.device.vendor_name}"
        )
    return result_dtype


def _cast_tensor(tensor, dtype):
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _validate_tensor_devices(*tensors):
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise RuntimeError("special_zeta expected all tensors to be on the same device")
    return device


def _tensors_overlap(left, right):
    try:
        return torch._C._overlaps(left, right)
    except AttributeError:
        return left is right


def _is_exact_alias(left, right):
    if left is right:
        return True
    if left.device != right.device or left.dtype != right.dtype:
        return False
    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
    )


def _prepare_out(out, shape, device, inputs):
    if out.device != device:
        raise RuntimeError(
            f"special_zeta expected out on {device}, but got {out.device}"
        )
    if not (out.is_floating_point() or out.is_complex()):
        raise RuntimeError(
            f"result type Float can't be cast to the desired output type {out.dtype}"
        )
    if has_internal_overlapping(out) == MemOverlap.Yes:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to tensor "
            "refers to a single memory location"
        )

    copy_back = out.is_complex()
    for tensor in inputs:
        if not _tensors_overlap(out, tensor):
            continue
        if not _is_exact_alias(out, tensor):
            raise RuntimeError(
                "unsupported operation: some elements of the input tensor and the "
                "written-to tensor refer to a single memory location"
            )
        copy_back = True

    if tuple(out.shape) != tuple(shape):
        if copy_back:
            raise RuntimeError(
                "special_zeta cannot resize an output that aliases an input"
            )
        out.resize_(shape)
    return copy_back


def special_zeta(x, q):
    logger.debug("GEMS SPECIAL_ZETA")
    _validate_tensor_devices(x, q)
    dtype = _promoted_dtype(x, q)
    x_compute = _cast_tensor(x, dtype)
    q_compute = _cast_tensor(q, dtype)
    shape = torch.broadcast_shapes(x_compute.shape, q_compute.shape)
    return _compute_result(
        x_compute,
        q_compute,
        shape,
        x.device,
        dtype,
        x_is_tensor=True,
        q_is_tensor=True,
    )


def special_zeta_out(x, q, out):
    logger.debug("GEMS SPECIAL_ZETA_OUT")
    device = _validate_tensor_devices(x, q)
    dtype = _promoted_dtype(x, q)
    shape = torch.broadcast_shapes(x.shape, q.shape)
    copy_back = _prepare_out(out, shape, device, (x, q))
    x_compute = _cast_tensor(x, dtype)
    q_compute = _cast_tensor(q, dtype)
    if copy_back:
        result = _compute_result(
            x_compute,
            q_compute,
            shape,
            device,
            dtype,
            x_is_tensor=True,
            q_is_tensor=True,
        )
        return out.copy_(result)
    return _launch_layout(
        x_compute,
        q_compute,
        out,
        dtype,
        x_is_tensor=True,
        q_is_tensor=True,
    )


def special_zeta_tensor_scalar(x, q):
    logger.debug("GEMS SPECIAL_ZETA_TENSOR_SCALAR")
    dtype = _promoted_dtype(x, q)
    x_compute = _cast_tensor(x, dtype)
    return _compute_result(
        x_compute,
        q,
        x.shape,
        x.device,
        dtype,
        x_is_tensor=True,
        q_is_tensor=False,
    )


def special_zeta_tensor_scalar_out(x, q, out):
    logger.debug("GEMS SPECIAL_ZETA_TENSOR_SCALAR_OUT")
    dtype = _promoted_dtype(x, q)
    copy_back = _prepare_out(out, x.shape, x.device, (x,))
    x_compute = _cast_tensor(x, dtype)
    if copy_back:
        result = _compute_result(
            x_compute,
            q,
            x.shape,
            x.device,
            dtype,
            x_is_tensor=True,
            q_is_tensor=False,
        )
        return out.copy_(result)
    return _launch_layout(
        x_compute,
        q,
        out,
        dtype,
        x_is_tensor=True,
        q_is_tensor=False,
    )


def special_zeta_scalar_tensor(x, q):
    logger.debug("GEMS SPECIAL_ZETA_SCALAR_TENSOR")
    dtype = _promoted_dtype(x, q)
    q_compute = _cast_tensor(q, dtype)
    return _compute_result(
        x,
        q_compute,
        q.shape,
        q.device,
        dtype,
        x_is_tensor=False,
        q_is_tensor=True,
    )


def special_zeta_scalar_tensor_out(x, q, out):
    logger.debug("GEMS SPECIAL_ZETA_SCALAR_TENSOR_OUT")
    dtype = _promoted_dtype(x, q)
    copy_back = _prepare_out(out, q.shape, q.device, (q,))
    q_compute = _cast_tensor(q, dtype)
    if copy_back:
        result = _compute_result(
            x,
            q_compute,
            q.shape,
            q.device,
            dtype,
            x_is_tensor=False,
            q_is_tensor=True,
        )
        return out.copy_(result)
    return _launch_layout(
        x,
        q_compute,
        out,
        dtype,
        x_is_tensor=False,
        q_is_tensor=True,
    )
