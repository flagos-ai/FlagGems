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

"""
Specialized mul for mthreads backend.

Problem:
  The common op (src/flag_gems/ops/mul.py) handles complex multiplication via
  mul_complex_broadcast_func, which calls _complex_output to create an empty
  complex64 tensor. This triggers empty_kernel with dtype=complex64, but mthreads
  Triton does not support complex64 in canonicalize_dtype (KeyError: 'complex64').

Fix:
  Decompose complex multiplication into real/imag part operations directly,
  avoiding the creation of complex64 output tensors via empty_kernel.

  - complex * complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
  - complex * real:    (a+bi)*c     = ac + bci
  - real * real:       use backend scalar and dense-row kernels
"""

import logging
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

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
    out_shape = _broadcast_shape(a_t, b_t)
    if out is None:
        return torch.empty(out_shape, device=a_t.device, dtype=a_t.dtype)
    if tuple(out.shape) != out_shape:
        raise RuntimeError(
            f"output with shape {tuple(out.shape)} cannot be broadcast to {out_shape}"
        )
    if out.dtype != a_t.dtype:
        raise RuntimeError(
            f"output dtype {out.dtype} does not match result dtype {a_t.dtype}"
        )
    if out.device != a_t.device:
        raise RuntimeError("output must be on the same device as inputs")
    return out


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


def _is_complex_op(A, B):
    """Check if either operand involves complex dtype."""
    if isinstance(A, torch.Tensor) and A.is_complex():
        return True
    if isinstance(B, torch.Tensor) and B.is_complex():
        return True
    if isinstance(A, complex) or isinstance(B, complex):
        return True
    return False


def _get_real_dtype(complex_dtype):
    """Map complex dtype to its corresponding real dtype."""
    if complex_dtype == torch.complex64:
        return torch.float32
    elif complex_dtype == torch.complex128:
        return torch.float64
    return torch.float32


def _complex_mul(A, B):
    """
    Complex multiplication decomposed into real/imag operations.

    Handles all combinations:
    1. complex tensor * complex tensor
    2. complex tensor * complex scalar
    3. complex tensor * real tensor
    4. complex tensor * real scalar (int/float)
    5. real tensor * complex tensor (symmetric cases)

    All paths avoid creating complex64 tensors via torch.empty/torch.tensor on
    device, which would trigger the unsupported empty_kernel path on mthreads.
    Instead, we decompose into real-valued arithmetic on the real/imag parts.
    """
    # Verify at least one operand is a tensor
    if not isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        raise TypeError("At least one operand must be a tensor")

    # Determine result complex dtype
    if isinstance(A, torch.Tensor) and A.is_complex():
        result_dtype = A.dtype
    elif isinstance(B, torch.Tensor) and B.is_complex():
        result_dtype = B.dtype
    else:
        result_dtype = torch.complex64

    real_dtype = _get_real_dtype(result_dtype)

    # Extract real/imag parts for A
    if isinstance(A, torch.Tensor) and A.is_complex():
        a_view = torch.view_as_real(A.to(dtype=result_dtype))
        ar, ai = a_view[..., 0], a_view[..., 1]
    elif isinstance(A, torch.Tensor):
        # Real tensor: imag part is zero
        ar = A.to(dtype=real_dtype)
        ai = torch.zeros_like(ar)
    elif isinstance(A, complex):
        # Complex scalar: extract real/imag as Python floats, broadcast later
        ar = A.real
        ai = A.imag
    else:
        # Real scalar (int/float): imag is zero
        ar = float(A)
        ai = 0.0

    # Extract real/imag parts for B
    if isinstance(B, torch.Tensor) and B.is_complex():
        b_view = torch.view_as_real(B.to(dtype=result_dtype))
        br, bi = b_view[..., 0], b_view[..., 1]
    elif isinstance(B, torch.Tensor):
        # Real tensor: imag part is zero
        br = B.to(dtype=real_dtype)
        bi = torch.zeros_like(br)
    elif isinstance(B, complex):
        # Complex scalar
        br = B.real
        bi = B.imag
    else:
        # Real scalar (int/float): imag is zero
        br = float(B)
        bi = 0.0

    # Optimized path: complex * real (imag part of one operand is zero)
    # (a+bi)*c = ac + bci  — avoids unnecessary multiplications
    if not isinstance(A, complex) and isinstance(A, (int, float)):
        # A is real scalar, B must be complex tensor
        out_r = ar * br - ai * bi
        out_i = ar * bi + ai * br
        out = torch.stack((out_r, out_i), dim=-1)
        return torch.view_as_complex(out.contiguous()).to(result_dtype)

    if not isinstance(B, complex) and isinstance(B, (int, float)):
        # B is real scalar: (a+bi)*c = ac + bci
        # ar, ai are tensors; br is float, bi is 0.0
        out_r = ar * br
        out_i = ai * br
        out = torch.stack((out_r, out_i), dim=-1)
        return torch.view_as_complex(out.contiguous()).to(result_dtype)

    if isinstance(B, torch.Tensor) and not B.is_complex():
        # B is real tensor: (a+bi)*c = ac + bci
        out_r = ar * br
        out_i = ai * br
        out = torch.stack((out_r, out_i), dim=-1)
        return torch.view_as_complex(out.contiguous()).to(result_dtype)

    if isinstance(A, torch.Tensor) and not A.is_complex():
        # A is real tensor: c*(b+di) = cb + cdi
        out_r = ar * br
        out_i = ar * bi
        out = torch.stack((out_r, out_i), dim=-1)
        return torch.view_as_complex(out.contiguous()).to(result_dtype)

    # General case: complex * complex
    # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    out_r = ar * br - ai * bi
    out_i = ar * bi + ai * br
    out = torch.stack((out_r, out_i), dim=-1)
    return torch.view_as_complex(out.contiguous()).to(result_dtype)


def mul(A, B, *, out=None):
    """
    mthreads specialized mul.

    For complex operations, decomposes into real/imag part arithmetic to avoid
    triggering empty_kernel with complex64 dtype (unsupported by mthreads Triton).
    For real*real, uses the local scalar and broadcast implementations.
    """
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        result = _try_contiguous(A, B, out)
        if result is not None:
            return result
    logger.debug("GEMS_MTHREADS MUL")
    if isinstance(A, torch.Tensor) or isinstance(B, torch.Tensor):
        if _is_complex_op(A, B):
            result = _complex_mul(A, B)
            if out is not None:
                out.copy_(result)
                return out
            return result
        return mul_broadcast_func(A, B, out=out)
    return torch.tensor(A * B)


def mul_(A, B):
    """
    mthreads specialized mul_ (inplace).

    Same complex decomposition strategy as mul().
    """
    logger.debug("GEMS_MTHREADS MUL_")
    if not isinstance(A, torch.Tensor):
        raise TypeError("mul_ expects the first argument to be a tensor")
    if _is_complex_op(A, B):
        result = _complex_mul(A, B)
        A.copy_(result)
        return A
    return mul_broadcast_func(A, B, out=A)


def _scalar_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in (
            (128, 1),
            (256, 1),
            (256, 2),
            (512, 2),
            (2048, 1),
        )
    ] + [triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3)]


def _row_configs():
    return _scalar_configs() + [
        triton.Config({"BLOCK_SIZE": block}, num_warps=4, num_stages=3)
        for block in (256, 512)
    ]


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


def _prune_mul_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    defaults = _row_configs() if "n_cols" in args else _scalar_configs()
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

    elements = args.get("n_elements", 0)
    outer = (
        args.get("dtype") == "float32"
        and 0 < args.get("n_cols", 0) <= 64
        and args.get("a_s0") == 1
        and args.get("a_s1") == 0
        and args.get("b_s0") == 0
        and args.get("b_s1") == 1
        and args.get("out_s0") == args.get("n_cols")
        and args.get("out_s1") == 1
    )
    if args.get("dtype") != "float32" or (
        "n_cols" in args and not args.get("DENSE_COLS", 0) and not outer
    ):
        blocks = (256, 512, 1024) if "n_cols" in args else (1024,)
        return [
            c
            for c in configs
            if c.num_warps == 4
            and c.num_stages == 3
            and c.kwargs["BLOCK_SIZE"] in blocks
        ]
    if outer:
        preferred = {(256, 2), (512, 2)} if elements < 4194304 else {(2048, 1)}
    elif "n_cols" in args:
        preferred = (
            {(256, 1), (256, 2)} if elements < 16777216 else {(128, 1), (256, 1)}
        )
    elif elements <= 2048:
        preferred = {(128, 1)}
    else:
        preferred = {(256, 1), (256, 2)}
    return [
        c
        for c in configs
        if c.num_stages == 1 and (c.kwargs["BLOCK_SIZE"], c.num_warps) in preferred
    ]


@triton.jit
def _mul_dense_row(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if n_elements % BLOCK_SIZE == 0:
        scale = tl.load(a_ptr + offsets // N_COLS)
        matrix = tl.load(b_ptr + offsets)
        tl.store(out_ptr + offsets, scale * matrix)
    else:
        mask = offsets < n_elements
        scale = tl.load(a_ptr + offsets // N_COLS, mask=mask, other=0)
        matrix = tl.load(b_ptr + offsets, mask=mask, other=0)
        tl.store(out_ptr + offsets, scale * matrix, mask=mask)


@triton.jit
def _mul_outer_broadcast(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    BLOCK_N: tl.constexpr = min(BLOCK_SIZE, triton.next_power_of_2(N_COLS))
    BLOCK_M: tl.constexpr = max(1, BLOCK_SIZE // BLOCK_N)
    COL_TILES: tl.constexpr = tl.cdiv(N_COLS, BLOCK_N)
    pid = tl.program_id(0)
    rows = (pid // COL_TILES) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = (pid % COL_TILES) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_elements // N_COLS
    col_mask = cols < N_COLS
    scale = tl.load(a_ptr + rows, row_mask, other=0)
    vector = tl.load(b_ptr + cols, col_mask, other=0)
    tl.store(
        out_ptr + rows[:, None] * N_COLS + cols[None, :],
        scale[:, None] * vector[None, :],
        row_mask[:, None] & col_mask[None, :],
    )


@libentry()
@libtuner(
    configs=_scalar_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _prune_mul_configs},
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
    configs=_row_configs(),
    key=["n_elements", "n_cols", "dtype", "DENSE_COLS"],
    strategy=["align32", "default", "default", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _prune_mul_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul_broadcast_2d_dense_row",
    flagtune_op_id="flaggems/mul",
    flagtune_variant="broadcast_2d",
    policy="flagtune",
)
@triton.jit
def mul_broadcast_2d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    n_cols: tl.constexpr,
    a_s0: tl.constexpr,
    a_s1: tl.constexpr,
    b_s0: tl.constexpr,
    b_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    DENSE_COLS: tl.constexpr = 0,
):
    if DENSE_COLS:
        _mul_dense_row(a_ptr, b_ptr, out_ptr, n_elements, DENSE_COLS, BLOCK_SIZE)
        return
    if (
        (dtype == "float32" or dtype == "bfloat16")
        and n_cols > 0
        and n_cols <= 64
        and a_s0 == 1
        and a_s1 == 0
        and b_s0 == 0
        and b_s1 == 1
        and out_s0 == n_cols
        and out_s1 == 1
    ):
        _mul_outer_broadcast(a_ptr, b_ptr, out_ptr, n_elements, n_cols, BLOCK_SIZE)
        return
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = offsets // n_cols
    col = offsets - row * n_cols
    a = tl.load(a_ptr + row * a_s0 + col * a_s1, mask=mask)
    b = tl.load(b_ptr + row * b_s0 + col * b_s1, mask=mask)
    value = a & b if IS_BOOL else a * b
    tl.store(out_ptr + row * out_s0 + col * out_s1, value, mask=mask)


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
    if not n_elements:
        return output
    cols = out_shape[1]
    dense_cols = 0
    outer_cols = 0
    if (
        dtype in (torch.float32, torch.bfloat16)
        and output.is_contiguous()
        and n_elements < 2**31
    ):
        if b_stride == (1, 0) and a_stride in ((cols, 1), (0, 1)):
            a_t, b_t = b_t, a_t
            a_stride, b_stride = b_stride, a_stride
        if a_stride == (1, 0) and b_stride == (cols, 1):
            dense_cols = cols
        elif a_stride == (1, 0) and b_stride == (0, 1) and 0 < cols <= 64:
            outer_cols = cols
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    if outer_cols:
        padded = triton.next_power_of_2(outer_cols)
        grid = lambda meta: (
            triton.cdiv(out_shape[0], max(1, meta["BLOCK_SIZE"] // padded))
            * triton.cdiv(outer_cols, min(meta["BLOCK_SIZE"], padded)),
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
            cols,
            a_s0=a_stride[0],
            a_s1=a_stride[1],
            b_s0=b_stride[0],
            b_s1=b_stride[1],
            out_s0=out_stride[0],
            out_s1=out_stride[1],
            dtype=_dtype_name(dtype),
            IS_BOOL=dtype is torch.bool,
            DENSE_COLS=dense_cols,
            **launch,
        )
    return output
