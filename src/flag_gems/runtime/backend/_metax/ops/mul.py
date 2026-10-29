# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX mul: scalar kernels and broadcast tiles."""

import copy
import logging
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

_DEVICE_NAME = runtime_device.name


def mul_get_configs():
    return [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ]


def mul_broadcast_get_configs():
    return [
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ]


def _contiguous_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in (
            (128, 1),
            (256, 1),
            (512, 2),
            (1024, 4),
            (1024, 8),
            (2048, 4),
            (4096, 4),
            (8192, 4),
        )
    ] + mul_get_configs()


def _prune_contiguous_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    defaults = {str(config) for config in _contiguous_configs()}
    # Expanded keeps every launch geometry; remove verified stage aliases.
    if any(str(config) not in defaults for config in configs):
        return configs
    if args.get("dtype") not in ("float32", "bfloat16"):
        original = {str(config) for config in mul_get_configs()}
        return [config for config in configs if str(config) in original]
    elements = args.get("n_elements", 0)
    if elements <= 16384:
        preferred = {(128, 1), (256, 1), (512, 2), (1024, 4)}
    elif elements < 1048576:
        preferred = {(512, 2), (1024, 4), (1024, 8)}
    else:
        preferred = {(1024, 4), (1024, 8), (2048, 4)}
    selected = [
        config
        for config in configs
        if (config.kwargs["BLOCK_SIZE"], config.num_warps) in preferred
    ]
    return selected or configs


@triton.jit
def _mul_contiguous_tile(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    block,
    BLOCK_SIZE: tl.constexpr,
):
    compute_dtype: tl.constexpr = (
        tl.float16 if x_ptr.dtype.element_ty == tl.float16 else tl.float32
    )
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if n_elements % BLOCK_SIZE == 0:
        x = tl.load(x_ptr + offsets).to(compute_dtype)
        y = tl.load(y_ptr + offsets).to(compute_dtype)
        tl.store(output_ptr + offsets, x * y)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask, other=0).to(compute_dtype)
        y = tl.load(y_ptr + offsets, mask, other=0).to(compute_dtype)
        tl.store(output_ptr + offsets, x * y, mask)


@libentry()
@libtuner(
    configs=mul_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x & y if IS_BOOL else x * y
    tl.store(output_ptr + offsets, out, mask=mask)


@libentry()
@libtuner(
    configs=_contiguous_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _prune_contiguous_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_contiguous_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    if (
        dtype == "float16" or dtype == "float32" or dtype == "bfloat16"
    ) and n_elements < 2147483648:
        _mul_contiguous_tile(
            x_ptr, y_ptr, output_ptr, n_elements, tl.program_id(0), BLOCK_SIZE
        )
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x & y if IS_BOOL else x * y
        tl.store(output_ptr + offsets, out, mask=mask)


@libentry()
@libtuner(
    configs=mul_broadcast_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_generic_nd_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    A_STRIDE: tl.constexpr,
    B_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    NDIM: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    linear = offsets
    a_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    b_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    for dim in tl.static_range(NDIM - 1, -1, -1):
        idx = linear % SHAPE[dim]
        linear = linear // SHAPE[dim]
        a_offsets += idx * A_STRIDE[dim]
        b_offsets += idx * B_STRIDE[dim]
        out_offsets += idx * OUT_STRIDE[dim]

    a = tl.load(a_ptr + a_offsets, mask=mask)
    b = tl.load(b_ptr + b_offsets, mask=mask)
    out = a & b if IS_BOOL else a * b
    tl.store(out_ptr + out_offsets, out, mask=mask)


@libentry()
@libtuner(
    configs=mul_broadcast_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_generic_nd_runtime_meta_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    meta_ptr,
    n_elements,
    NDIM: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    linear = offsets
    a_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    b_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    out_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    for dim in tl.static_range(NDIM - 1, -1, -1):
        shape_dim = tl.load(meta_ptr + dim)
        a_stride_dim = tl.load(meta_ptr + NDIM + dim)
        b_stride_dim = tl.load(meta_ptr + 2 * NDIM + dim)
        out_stride_dim = tl.load(meta_ptr + 3 * NDIM + dim)
        idx = linear % shape_dim
        linear = linear // shape_dim
        a_offsets += idx * a_stride_dim
        b_offsets += idx * b_stride_dim
        out_offsets += idx * out_stride_dim

    a = tl.load(a_ptr + a_offsets, mask=mask)
    b = tl.load(b_ptr + b_offsets, mask=mask)
    out = a & b if IS_BOOL else a * b
    tl.store(out_ptr + out_offsets, out, mask=mask)


@libentry()
@libtuner(
    configs=mul_broadcast_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_complex_generic_nd_kernel(
    ar_ptr,
    ai_ptr,
    br_ptr,
    bi_ptr,
    out_r_ptr,
    out_i_ptr,
    n_elements,
    SHAPE: tl.constexpr,
    AR_STRIDE: tl.constexpr,
    AI_STRIDE: tl.constexpr,
    BR_STRIDE: tl.constexpr,
    BI_STRIDE: tl.constexpr,
    OUT_R_STRIDE: tl.constexpr,
    OUT_I_STRIDE: tl.constexpr,
    NDIM: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    linear = offsets
    ar_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    ai_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    br_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    bi_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    out_r_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    out_i_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    for dim in tl.static_range(NDIM - 1, -1, -1):
        idx = linear % SHAPE[dim]
        linear = linear // SHAPE[dim]
        ar_offsets += idx * AR_STRIDE[dim]
        ai_offsets += idx * AI_STRIDE[dim]
        br_offsets += idx * BR_STRIDE[dim]
        bi_offsets += idx * BI_STRIDE[dim]
        out_r_offsets += idx * OUT_R_STRIDE[dim]
        out_i_offsets += idx * OUT_I_STRIDE[dim]

    ar = tl.load(ar_ptr + ar_offsets, mask=mask, other=0.0)
    ai = tl.load(ai_ptr + ai_offsets, mask=mask, other=0.0)
    br = tl.load(br_ptr + br_offsets, mask=mask, other=0.0)
    bi = tl.load(bi_ptr + bi_offsets, mask=mask, other=0.0)

    out_r = ar * br - ai * bi
    out_i = ar * bi + ai * br
    tl.store(out_r_ptr + out_r_offsets, out_r, mask=mask)
    tl.store(out_i_ptr + out_i_offsets, out_i, mask=mask)


@libentry()
@libtuner(
    configs=mul_broadcast_get_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit
def mul_complex_generic_nd_runtime_meta_kernel(
    ar_ptr,
    ai_ptr,
    br_ptr,
    bi_ptr,
    out_r_ptr,
    out_i_ptr,
    meta_ptr,
    n_elements,
    NDIM: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    linear = offsets
    ar_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    ai_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    br_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    bi_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    out_r_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    out_i_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)

    for dim in tl.static_range(NDIM - 1, -1, -1):
        shape_dim = tl.load(meta_ptr + dim)
        ar_stride_dim = tl.load(meta_ptr + NDIM + dim)
        ai_stride_dim = tl.load(meta_ptr + 2 * NDIM + dim)
        br_stride_dim = tl.load(meta_ptr + 3 * NDIM + dim)
        bi_stride_dim = tl.load(meta_ptr + 4 * NDIM + dim)
        out_r_stride_dim = tl.load(meta_ptr + 5 * NDIM + dim)
        out_i_stride_dim = tl.load(meta_ptr + 6 * NDIM + dim)
        idx = linear % shape_dim
        linear = linear // shape_dim
        ar_offsets += idx * ar_stride_dim
        ai_offsets += idx * ai_stride_dim
        br_offsets += idx * br_stride_dim
        bi_offsets += idx * bi_stride_dim
        out_r_offsets += idx * out_r_stride_dim
        out_i_offsets += idx * out_i_stride_dim

    ar = tl.load(ar_ptr + ar_offsets, mask=mask, other=0.0)
    ai = tl.load(ai_ptr + ai_offsets, mask=mask, other=0.0)
    br = tl.load(br_ptr + br_offsets, mask=mask, other=0.0)
    bi = tl.load(bi_ptr + bi_offsets, mask=mask, other=0.0)

    out_r = ar * br - ai * bi
    out_i = ar * bi + ai * br
    tl.store(out_r_ptr + out_r_offsets, out_r, mask=mask)
    tl.store(out_i_ptr + out_i_offsets, out_i, mask=mask)


def _select_device(a, b):
    for value in (a, b):
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _as_tensor(value, *, device, dtype):
    if isinstance(value, torch.Tensor):
        if value.device != device or value.dtype != dtype:
            return value.to(device=device, dtype=dtype)
        return value
    return torch.tensor(value, device=device, dtype=dtype)


def _dtype_name(dtype):
    return str(dtype).split(".")[-1]


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


def _broadcast_shape(a_t, b_t):
    return tuple(torch.broadcast_shapes(tuple(a_t.shape), tuple(b_t.shape)))


def _broadcasted_stride(shape, stride, out_shape):
    if not out_shape:
        return ()
    offset = len(out_shape) - len(shape)
    result = []
    for out_dim, out_size in enumerate(out_shape):
        in_dim = out_dim - offset
        if in_dim < 0:
            result.append(0)
            continue
        size = shape[in_dim]
        result.append(0 if size == 1 and out_size != 1 else stride[in_dim])
    return tuple(result)


def _real_output(a_t, b_t, out=None):
    return _output(
        _broadcast_shape(a_t, b_t), device=a_t.device, dtype=a_t.dtype, out=out
    )


def _can_use_contiguous_tensor_tensor(a_t, b_t, out):
    return (
        tuple(a_t.shape) == tuple(b_t.shape) == tuple(out.shape)
        and a_t.is_contiguous()
        and b_t.is_contiguous()
        and out.is_contiguous()
        and not a_t.is_complex()
        and not b_t.is_complex()
    )


def _real_layout(out_shape, a_t, b_t, out):
    a_stride = _broadcasted_stride(tuple(a_t.shape), tuple(a_t.stride()), out_shape)
    b_stride = _broadcasted_stride(tuple(b_t.shape), tuple(b_t.stride()), out_shape)
    out_stride = tuple(out.stride()) if out_shape else ()
    return a_stride, b_stride, out_stride


def _launch_contiguous_tensor_tensor(a_t, b_t, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        if (
            n_elements < 2**31
            and not torch._C._is_alias_of(output, a_t)
            and not torch._C._is_alias_of(output, b_t)
        ):
            mul_contiguous_kernel[grid](
                a_t,
                b_t,
                output,
                n_elements,
                dtype=_dtype_name(dtype),
                IS_BOOL=dtype is torch.bool,
            )
            return output
        mul_kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=dtype is torch.bool,
        )
    return output


def _try_contiguous(a, b, out=None):
    if not (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and not a.is_complex()
        and a.device.type == _DEVICE_NAME
        and a.device == b.device
        and a.dtype == b.dtype
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and a.numel() < 2**31
    ):
        return None
    if out is not None and (
        torch._C._is_alias_of(out, a) or torch._C._is_alias_of(out, b)
    ):
        return None
    output = torch.empty_like(a) if out is None else _real_output(a, b, out=out)
    if not output.is_contiguous():
        return None
    return _launch_contiguous_tensor_tensor(a, b, output, a.dtype)


def _launch_generic(a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    shape = out_shape or (1,)
    ndim = len(shape)
    a_stride = a_stride or (0,)
    b_stride = b_stride or (0,)
    out_stride = out_stride or (0,)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    if _needs_runtime_meta_for_constexpr_tuple():
        meta = torch.tensor(
            shape + a_stride + b_stride + out_stride,
            device=output.device,
            dtype=torch.int64,
        )
        with torch_device_fn.device(output.device):
            mul_generic_nd_runtime_meta_kernel[grid](
                a_t,
                b_t,
                output,
                meta,
                n_elements,
                NDIM=ndim,
                dtype=_dtype_name(dtype),
                IS_BOOL=dtype is torch.bool,
            )
        return output

    with torch_device_fn.device(output.device):
        mul_generic_nd_kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            SHAPE=shape,
            A_STRIDE=a_stride,
            B_STRIDE=b_stride,
            OUT_STRIDE=out_stride,
            NDIM=ndim,
            dtype=_dtype_name(dtype),
            IS_BOOL=dtype is torch.bool,
        )
    return output


def mul_broadcast_func(a, b, out=None):
    if not (
        isinstance(a, (torch.Tensor, Number)) and isinstance(b, (torch.Tensor, Number))
    ):
        raise TypeError("mul expects tensor or scalar inputs")

    device = _select_device(a, b)
    if device.type != _DEVICE_NAME:
        raise RuntimeError(f"mul expects {_DEVICE_NAME} inputs, got {device}")
    dtype = torch.result_type(a, b)

    # Multiplication is commutative; normalize tensor/scalar order once.
    if not isinstance(a, torch.Tensor):
        a, b = b, a
    if not isinstance(b, (torch.Tensor, complex)):
        a_t = _as_tensor(a, device=device, dtype=dtype)
        output = _real_output(a_t, a_t, out=out)
        if a_t.is_contiguous() and output.is_contiguous() and not a_t.is_complex():
            return _launch_scalar(a_t, b, output, dtype)

    a_t = _as_tensor(a, device=device, dtype=dtype)
    b_t = _as_tensor(b, device=device, dtype=dtype)
    output = _real_output(a_t, b_t, out=out)

    if _can_use_contiguous_tensor_tensor(a_t, b_t, output):
        return _launch_contiguous_tensor_tensor(a_t, b_t, output, dtype)

    out_shape = tuple(output.shape)
    a_stride, b_stride, out_stride = _real_layout(out_shape, a_t, b_t, output)
    if len(out_shape) == 2:
        return _launch_2d_broadcast(
            a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
        )
    return _launch_generic(
        a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
    )


def _complex_parts(value, *, device, complex_dtype):
    tensor = _as_tensor(value, device=device, dtype=complex_dtype)
    real_view = torch.view_as_real(tensor)
    return real_view[..., 0], real_view[..., 1]


def _output(out_shape, *, device, dtype, out=None):
    if out is None:
        return torch.empty(out_shape, device=device, dtype=dtype)
    if tuple(out.shape) != out_shape:
        raise RuntimeError(
            f"output with shape {tuple(out.shape)} cannot be broadcast to {out_shape}"
        )
    if out.dtype != dtype:
        raise RuntimeError(
            f"output dtype {out.dtype} does not match result dtype {dtype}"
        )
    if out.device != device:
        raise RuntimeError("output must be on the same device as inputs")
    return out


def _complex_layout(out_shape, tensors, out_r, out_i):
    strides = [
        _broadcasted_stride(tuple(tensor.shape), tuple(tensor.stride()), out_shape)
        for tensor in tensors
    ]
    return (
        strides,
        tuple(out_r.stride()) if out_shape else (),
        tuple(out_i.stride()) if out_shape else (),
    )


def _launch_complex_generic(
    ar, ai, br, bi, output, out_shape, strides, out_r_stride, out_i_stride, dtype
):
    n_elements = volume(out_shape)
    if n_elements == 0:
        return output
    out_view = torch.view_as_real(output)
    out_r = out_view[..., 0]
    out_i = out_view[..., 1]
    shape = out_shape or (1,)
    ndim = len(shape)
    strides = [stride or (0,) for stride in strides]
    out_r_stride = out_r_stride or (0,)
    out_i_stride = out_i_stride or (0,)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    if _needs_runtime_meta_for_constexpr_tuple():
        meta = torch.tensor(
            shape
            + strides[0]
            + strides[1]
            + strides[2]
            + strides[3]
            + out_r_stride
            + out_i_stride,
            device=output.device,
            dtype=torch.int64,
        )
        with torch_device_fn.device(output.device):
            mul_complex_generic_nd_runtime_meta_kernel[grid](
                ar,
                ai,
                br,
                bi,
                out_r,
                out_i,
                meta,
                n_elements,
                NDIM=ndim,
                dtype=_dtype_name(dtype),
            )
        return output

    with torch_device_fn.device(output.device):
        mul_complex_generic_nd_kernel[grid](
            ar,
            ai,
            br,
            bi,
            out_r,
            out_i,
            n_elements,
            SHAPE=shape,
            AR_STRIDE=strides[0],
            AI_STRIDE=strides[1],
            BR_STRIDE=strides[2],
            BI_STRIDE=strides[3],
            OUT_R_STRIDE=out_r_stride,
            OUT_I_STRIDE=out_i_stride,
            NDIM=ndim,
            dtype=_dtype_name(dtype),
        )
    return output


def mul_complex_broadcast_func(a, b, out=None):
    device = _select_device(a, b)
    if device.type != _DEVICE_NAME:
        raise RuntimeError(f"mul expects {_DEVICE_NAME} inputs, got {device}")
    dtype = torch.result_type(a, b)
    ar, ai = _complex_parts(a, device=device, complex_dtype=dtype)
    br, bi = _complex_parts(b, device=device, complex_dtype=dtype)
    out_shape = tuple(
        torch.broadcast_shapes(
            tuple(ar.shape), tuple(ai.shape), tuple(br.shape), tuple(bi.shape)
        )
    )
    output = _output(out_shape, device=device, dtype=dtype, out=out)
    out_view = torch.view_as_real(output)
    strides, out_r_stride, out_i_stride = _complex_layout(
        out_shape, [ar, ai, br, bi], out_view[..., 0], out_view[..., 1]
    )
    return _launch_complex_generic(
        ar, ai, br, bi, output, out_shape, strides, out_r_stride, out_i_stride, dtype
    )


@triton.jit
def _broadcast_tiled(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    N: tl.constexpr,
    TILE_COLS: tl.constexpr,
    OUTER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Layout validation and launch parameters belong to the caller's backend.
    BLOCK_N: tl.constexpr = min(BLOCK_SIZE, TILE_COLS)
    BLOCK_M: tl.constexpr = max(1, BLOCK_SIZE // TILE_COLS)
    COL_TILES: tl.constexpr = tl.cdiv(N, BLOCK_N)
    pid = tl.program_id(0)
    rows = (pid // COL_TILES) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = (pid % COL_TILES) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_elements // N
    col_mask = cols < N
    a = tl.load(a_ptr + rows, row_mask, other=0)
    if OUTER:
        b = tl.load(b_ptr + cols, col_mask, other=0)
        value = a[:, None] * b[None, :]
    else:
        b = tl.load(
            b_ptr + rows[:, None] * N + cols[None, :],
            row_mask[:, None] & col_mask[None, :],
            other=0,
        )
        value = a[:, None] * b
    tl.store(
        out_ptr + rows[:, None] * N + cols[None, :],
        value,
        row_mask[:, None] & col_mask[None, :],
    )


def scalar_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in ((128, 1), (1024, 1), (2048, 1), (2048, 2), (2048, 4))
    ] + [triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3)]


def broadcast_configs():
    configs = [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in (
            (128, 1),
            (256, 2),
            (512, 2),
            (512, 4),
            (1024, 1),
            (1024, 2),
            (2048, 1),
            (2048, 2),
            (4096, 1),
            (8192, 4),
        )
    ]
    configs.extend(
        triton.Config({"BLOCK_SIZE": block}, num_warps=4, num_stages=3)
        for block in (256, 512, 1024)
    )
    return configs


def _canonical_stage_configs(configs):
    """Keep stage 1 for equivalent straight-line scalar/broadcast kernels."""

    def key(config):
        return tuple(
            (name, repr(value))
            for name, value in sorted(config.all_kwargs().items())
            if name != "num_stages"
        )

    eligible = [
        config
        for config in configs
        if config.pre_hook is None and getattr(config, "ir_override", None) is None
    ]
    stage_one = {key(config) for config in eligible if config.num_stages == 1}
    return [
        config
        for config in configs
        if config.pre_hook is not None
        or getattr(config, "ir_override", None) is not None
        or not 1 < config.num_stages <= 8
        or key(config) not in stage_one
    ]


def prune_configs(configs, named_args, **kwargs):
    """Apply measured preferences only to the declared default pool."""
    args = {**named_args, **kwargs}
    defaults = broadcast_configs() if "n_cols" in args else scalar_configs()
    default_signatures = {str(config) for config in defaults}
    # Cost Model scores each verified launch geometry once.
    # Expanded/Train still collect the complete declared config space.
    if any(str(config) not in default_signatures for config in configs):
        return (
            _canonical_stage_configs(configs)
            if args.get("dtype") in ("float32", "bfloat16")
            and runtime.resolve_tuning_mode("mul", supports_cost_model=True)
            is runtime.TuningMode.COST_MODEL
            else configs
        )
    if args.get("dtype") == "bfloat16":
        return configs

    if args.get("dtype") not in ("float32", "bfloat16"):
        blocks = (256, 512, 1024) if "n_cols" in args else (1024,)
        return [
            c
            for c in configs
            if c.num_warps == 4
            and c.num_stages == 3
            and c.kwargs["BLOCK_SIZE"] in blocks
        ]

    candidates = [
        copy.deepcopy(config)
        for config in configs
        if config.num_warps in (1, 2, 4, 8)
        and (
            config.num_stages == 1
            or (
                config.num_stages == 3
                and config.num_warps == 4
                and config.kwargs["BLOCK_SIZE"] in (256, 512, 1024)
            )
        )
    ]
    cols = args.get("METAX_N_COLS", 0)
    if cols:
        if args["b_s0"] == 0:
            preferred = (
                {(1024, 1, 1), (2048, 2, 1)}
                if args.get("n_elements", 0) <= 1048576
                else {(2048, 1, 1), (8192, 4, 1)}
            )
        else:
            preferred = {(1024, 2, 1), (2048, 1, 1), (2048, 2, 1)}
        candidates = [
            config
            for config in candidates
            if (
                config.kwargs["BLOCK_SIZE"],
                config.num_warps,
                config.num_stages,
            )
            in preferred
        ]
    elif "n_cols" not in args and args.get("dtype") == "float32":
        elements = args.get("n_elements", 0)
        if elements <= 2048:
            preferred = {(128, 1, 1)}
        elif elements <= 131072:
            preferred = {(1024, 1, 1), (2048, 2, 1)}
        elif elements < 1048576:
            preferred = {(2048, 2, 1), (2048, 4, 1)}
        else:
            preferred = {(2048, 1, 1)}
        if preferred:
            candidates = [
                c
                for c in candidates
                if (c.kwargs["BLOCK_SIZE"], c.num_warps, c.num_stages) in preferred
            ]
    elif "n_cols" in args:
        # Small and generic-stride broadcasts retain the original candidates.
        candidates = [
            config
            for config in candidates
            if config.num_stages == 3 and config.num_warps == 4
        ]
    return candidates


@libentry()
@libtuner(
    configs=scalar_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": prune_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
    flagtune_op_id="flaggems/mul",
    flagtune_variant="scalar",
    policy="flagtune",
)
@triton.jit
def mul_scalar_kernel(
    x_ptr,
    output_ptr,
    scalar,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    if dtype == "float32" and n_elements >= 1048576:
        offsets = (
            tl.program_id(0) * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE // 4)[:, None] * 4
            + tl.arange(0, 4)[None, :]
        )
    else:
        offsets = tl.arange(0, BLOCK_SIZE)
        if n_elements > BLOCK_SIZE:
            offsets += tl.program_id(0) * BLOCK_SIZE
    # Exact tiles need no per-element bounds checks. All launchers use ceildiv
    # grids, so only the final partial tile requires a mask.
    if n_elements % BLOCK_SIZE == 0:
        x = tl.load(x_ptr + offsets)
        value = x & scalar if IS_BOOL else x * scalar
        tl.store(output_ptr + offsets, value)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        value = x & scalar if IS_BOOL else x * scalar
        tl.store(output_ptr + offsets, value, mask=mask)


@libentry()
@libtuner(
    configs=broadcast_configs(),
    key=[
        "n_elements",
        "n_cols",
        "dtype",
        "a_s0",
        "a_s1",
        "b_s0",
        "b_s1",
        "out_s0",
        "out_s1",
        "METAX_N_COLS",
    ],
    strategy=[
        "align32",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
        "default",
    ],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": prune_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul_broadcast_2d_metax",
    flagtune_op_id="flaggems/mul",
    flagtune_variant="broadcast_2d",
    policy="flagtune",
)
@triton.jit
def mul_broadcast_2d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    n_cols,
    a_s0: tl.constexpr,
    a_s1: tl.constexpr,
    b_s0: tl.constexpr,
    b_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    METAX_N_COLS: tl.constexpr = 0,
    METAX_TILE_COLS: tl.constexpr = 0,
):
    if METAX_N_COLS:
        _broadcast_tiled(
            a_ptr,
            b_ptr,
            out_ptr,
            n_elements,
            METAX_N_COLS,
            METAX_TILE_COLS,
            b_s0 == 0,
            BLOCK_SIZE,
        )
        return
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = offsets // n_cols
    col = offsets - row * n_cols
    a = tl.load(a_ptr + row * a_s0 + col * a_s1, mask=mask)
    b = tl.load(b_ptr + row * b_s0 + col * b_s1, mask=mask)
    out = a & b if IS_BOOL else a * b
    tl.store(out_ptr + row * out_s0 + col * out_s1, out, mask=mask)


def _launch_scalar(tensor, scalar, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        kernel = mul_scalar_kernel
        launch = {}
        if any(torch._C._is_alias_of(output, x) for x in (tensor,)):
            kernel = kernel.fn.fn
            launch = dict(BLOCK_SIZE=1024, num_warps=4, num_stages=3)
        kernel[grid](
            tensor,
            output,
            scalar,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=dtype is torch.bool,
            **launch,
        )
    return output


def _launch_2d_broadcast(
    a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    metax_n_cols = 0
    metax_tile_cols = 0
    if dtype in (torch.float32, torch.bfloat16) and output.is_contiguous():
        rows, cols = out_shape
        # Reversed row and outer broadcasts share the same dense tile kernel.
        if b_stride == (1, 0) and a_stride in ((cols, 1), (0, 1)):
            a_t, b_t = (b_t, a_t)
            a_stride, b_stride = (b_stride, a_stride)
        if (
            n_elements < 2**31
            and a_stride == (1, 0)
            and (b_stride in ((cols, 1), (0, 1)))
            and (
                (b_stride[0] == 0 and 1 < cols <= 64)
                or (b_stride[0] != 0 and 32 < rows < 1024 and 1 < cols <= 4096)
            )
        ):
            # These C550 thresholds regress for some shapes on other vendors.
            metax_n_cols = cols
            metax_tile_cols = triton.next_power_of_2(cols)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    if metax_n_cols:
        grid = lambda meta: (
            triton.cdiv(out_shape[0], max(1, meta["BLOCK_SIZE"] // metax_tile_cols))
            * triton.cdiv(metax_n_cols, min(meta["BLOCK_SIZE"], metax_tile_cols)),
        )
    with torch_device_fn.device(output.device):
        kernel = mul_broadcast_2d_kernel
        launch = {}
        if any(torch._C._is_alias_of(output, x) for x in (a_t, b_t)):
            kernel = kernel.fn.fn
            launch = dict(BLOCK_SIZE=1024, num_warps=4, num_stages=3)
        kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            out_shape[1],
            a_s0=a_stride[0],
            a_s1=a_stride[1],
            b_s0=b_stride[0],
            b_s1=b_stride[1],
            out_s0=out_stride[0],
            out_s1=out_stride[1],
            dtype=_dtype_name(dtype),
            IS_BOOL=dtype is torch.bool,
            METAX_N_COLS=metax_n_cols,
            METAX_TILE_COLS=metax_tile_cols,
            **launch,
        )
    return output


def mul(A, B, *, out=None):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        result = _try_contiguous(A, B, out)
        if result is not None:
            return result
    logger.debug("GEMS MUL")
    if isinstance(A, torch.Tensor) or isinstance(B, torch.Tensor):
        if (
            (isinstance(A, torch.Tensor) and A.is_complex())
            or (isinstance(B, torch.Tensor) and B.is_complex())
            or isinstance(A, complex)
            or isinstance(B, complex)
        ):
            return mul_complex_broadcast_func(A, B, out=out)
        return mul_broadcast_func(A, B, out=out)
    return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS MUL_")
    if not isinstance(A, torch.Tensor):
        raise TypeError("mul_ expects the first argument to be a tensor")
    dtype = torch.result_type(A, B)
    if dtype != A.dtype:
        raise RuntimeError(
            f"result type {dtype} cannot be cast to inplace dtype {A.dtype}"
        )
    if (
        A.is_complex()
        or (isinstance(B, torch.Tensor) and B.is_complex())
        or isinstance(B, complex)
    ):
        return mul_complex_broadcast_func(A, B, out=A)
    return mul_broadcast_func(A, B, out=A)
