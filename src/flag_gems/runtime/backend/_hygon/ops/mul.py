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
from contextlib import nullcontext
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import flagtune_enabled, torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.device_info import get_sm_count
from flag_gems.utils.shape_utils import volume

# Local kernels and fallback paths keep this backend independent of ops.mul.

_DEVICE_NAME = runtime_device.name

_USE_CUDA_GRAPH = True

_USE_HOPPER_GRID = False

_USE_CONTIGUOUS_FAST_PATH = True

_CONTIGUOUS_DTYPES = (torch.float32, torch.bfloat16)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


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
    if not _USE_CONTIGUOUS_FAST_PATH:
        return mul_get_configs()
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
    # Expanded retains its full YAML domain, including the original default.
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
        and (config.num_stages == 1 or runtime_device.vendor_name != "nvidia")
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
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if n_elements % BLOCK_SIZE == 0:
        x = tl.load(x_ptr + offsets).to(tl.float32)
        y = tl.load(y_ptr + offsets).to(tl.float32)
        tl.store(output_ptr + offsets, x * y)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask, other=0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask, other=0).to(tl.float32)
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
    use_cuda_graph=_USE_CONTIGUOUS_FAST_PATH,
    prune_configs_by={"early_config_prune": _prune_contiguous_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
)
@triton.jit(
    do_not_specialize=(
        ["x_ptr", "y_ptr", "output_ptr"]
        if runtime_device.vendor_name == "hygon"
        else []
    )
)
def mul_contiguous_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    NUM_SMS: tl.constexpr = 0,
):
    if (dtype == "float32" or dtype == "bfloat16") and n_elements < 2147483648:
        # NUM_SMS is nonzero only for the validated Hopper BF16 launch policy.
        if NUM_SMS > 0 and tl.cdiv(n_elements, BLOCK_SIZE) > NUM_SMS * 8:
            for block in range(
                tl.program_id(0), tl.cdiv(n_elements, BLOCK_SIZE), tl.num_programs(0)
            ):
                _mul_contiguous_tile(
                    x_ptr, y_ptr, output_ptr, n_elements, block, BLOCK_SIZE
                )
        else:
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


_hygon_contiguous_default = (
    libentry()(mul_contiguous_kernel.fn.fn)
    if runtime_device.vendor_name == "hygon"
    else None
)


def _hygon_contiguous_config(n_elements, dtype):
    if n_elements <= 4096:
        if dtype == torch.bfloat16:
            return 128, 1
        return (1024, 4) if n_elements <= 1024 else (256, 1)
    if n_elements <= 16384:
        return 512, 2
    if n_elements < 1048576:
        return 1024, 8
    if n_elements < 4194304:
        return 2048, 4
    return (1024, 4) if dtype == torch.bfloat16 else (1024, 8)


class _FallbackKernels:
    """Keep legacy dtype paths separate from the public tuning variants."""

    @libentry()
    @libtuner(
        configs=mul_get_configs(),
        key=["n_elements", "dtype"],
        strategy=["align32", "default"],
        warmup=5,
        rep=5,
        use_cuda_graph=_USE_CUDA_GRAPH,
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
        n_elements,
        dtype: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        IS_BOOL: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        out = x & scalar if IS_BOOL else x * scalar
        tl.store(output_ptr + offsets, out, mask=mask)

    @libentry()
    @libtuner(
        configs=mul_broadcast_get_configs(),
        key=["n_elements", "n_cols", "dtype"],
        strategy=["align32", "default", "default"],
        warmup=5,
        rep=5,
        use_cuda_graph=_USE_CUDA_GRAPH,
        flagtune_op_name="mul",
        flagtune_expand_op_name="mul_broadcast_2d",
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
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        row = offsets // n_cols
        col = offsets - row * n_cols
        a = tl.load(a_ptr + row * a_s0 + col * a_s1, mask=mask)
        b = tl.load(b_ptr + row * b_s0 + col * b_s1, mask=mask)
        out = a & b if IS_BOOL else a * b
        tl.store(out_ptr + row * out_s0 + col * out_s1, out, mask=mask)


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


def _is_tensor_or_number(value) -> bool:
    return isinstance(value, torch.Tensor) or isinstance(value, Number)


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


def _result_dtype(a, b):
    return torch.result_type(a, b)


def _dtype_name(dtype):
    return str(dtype).split(".")[-1]


def _is_bool_dtype(dtype):
    return dtype is torch.bool


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


def _fallback_broadcast_shape(a_t, b_t):
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


def _fallback_real_output(a_t, b_t, out=None):
    out_shape = _fallback_broadcast_shape(a_t, b_t)
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


def _can_use_contiguous_scalar(tensor, out):
    return tensor.is_contiguous() and out.is_contiguous() and not tensor.is_complex()


def _real_layout(out_shape, a_t, b_t, out):
    a_stride = _broadcasted_stride(tuple(a_t.shape), tuple(a_t.stride()), out_shape)
    b_stride = _broadcasted_stride(tuple(b_t.shape), tuple(b_t.stride()), out_shape)
    out_stride = tuple(out.stride()) if out_shape else ()
    return a_stride, b_stride, out_stride


def _can_use_2d_broadcast(out_shape):
    return len(out_shape) == 2


def _launch_contiguous_tensor_tensor_fast(a_t, b_t, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    guard = (
        nullcontext()
        if runtime_device.vendor_name == "hygon"
        and torch_device_fn.current_device() == output.device.index
        else torch_device_fn.device(output.device)
    )
    with guard:
        if (
            _USE_CONTIGUOUS_FAST_PATH
            and dtype in _CONTIGUOUS_DTYPES
            and n_elements < 2**31
            and not torch._C._is_alias_of(output, a_t)
            and not torch._C._is_alias_of(output, b_t)
        ):
            num_sms = 0
            if (
                _USE_HOPPER_GRID
                and dtype == torch.bfloat16
                and 1048576 <= n_elements <= 4194304
            ):
                num_sms = get_sm_count()

            def contiguous_grid(meta):
                blocks = triton.cdiv(n_elements, meta["BLOCK_SIZE"])
                return (min(blocks, num_sms * 8) if num_sms else blocks,)

            if (
                _hygon_contiguous_default is not None
                and n_elements <= 4194304
                and mul_contiguous_kernel.fn._run_mode == "normal"
                and not flagtune_enabled("mul")
            ):
                block, warps = _hygon_contiguous_config(n_elements, dtype)
                _hygon_contiguous_default[contiguous_grid](
                    a_t,
                    b_t,
                    output,
                    n_elements,
                    dtype=_dtype_name(dtype),
                    IS_BOOL=False,
                    NUM_SMS=num_sms,
                    BLOCK_SIZE=block,
                    num_warps=warps,
                    num_stages=1,
                )
                return output
            mul_contiguous_kernel[contiguous_grid](
                a_t,
                b_t,
                output,
                n_elements,
                dtype=_dtype_name(dtype),
                IS_BOOL=False,
                NUM_SMS=num_sms,
            )
            return output
        mul_kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=_is_bool_dtype(dtype),
        )
    return output


def _try_contiguous(a, b, out=None):
    if not (
        _USE_CONTIGUOUS_FAST_PATH
        and isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.dtype in _CONTIGUOUS_DTYPES
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
    output = (
        torch.empty_like(a) if out is None else _fallback_real_output(a, b, out=out)
    )
    if not output.is_contiguous():
        return None
    return _launch_contiguous_tensor_tensor_fast(a, b, output, a.dtype)


def _fallback_launch_scalar(tensor, scalar, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        _FallbackKernels.mul_scalar_kernel[grid](
            tensor,
            output,
            scalar,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=_is_bool_dtype(dtype),
        )
    return output


def _fallback_launch_2d_broadcast(
    a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        _FallbackKernels.mul_broadcast_2d_kernel[grid](
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
            IS_BOOL=_is_bool_dtype(dtype),
        )
    return output


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
                IS_BOOL=_is_bool_dtype(dtype),
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
            IS_BOOL=_is_bool_dtype(dtype),
        )
    return output


def _fallback_mul_broadcast_func(a, b, out=None):
    if not (_is_tensor_or_number(a) and _is_tensor_or_number(b)):
        raise TypeError("mul expects tensor or scalar inputs")

    device = _select_device(a, b)
    if device.type != _DEVICE_NAME:
        if out is not None:
            return torch.ops.aten.mul.out.redispatch(_FALLBACK_KEYSET, a, b, out=out)
        return torch.ops.aten.mul.Tensor.redispatch(_FALLBACK_KEYSET, a, b)

    dtype = _result_dtype(a, b)

    if (
        isinstance(a, torch.Tensor)
        and not isinstance(b, torch.Tensor)
        and not isinstance(b, complex)
    ):
        a_t = _as_tensor(a, device=device, dtype=dtype)
        output = _fallback_real_output(a_t, a_t.new_empty(()), out=out)
        if _can_use_contiguous_scalar(a_t, output):
            return _fallback_launch_scalar(a_t, b, output, dtype)

    if (
        isinstance(b, torch.Tensor)
        and not isinstance(a, torch.Tensor)
        and not isinstance(a, complex)
    ):
        b_t = _as_tensor(b, device=device, dtype=dtype)
        output = _fallback_real_output(b_t.new_empty(()), b_t, out=out)
        if _can_use_contiguous_scalar(b_t, output):
            return _fallback_launch_scalar(b_t, a, output, dtype)

    a_t = _as_tensor(a, device=device, dtype=dtype)
    b_t = _as_tensor(b, device=device, dtype=dtype)
    output = _fallback_real_output(a_t, b_t, out=out)

    if _can_use_contiguous_tensor_tensor(a_t, b_t, output):
        return _launch_contiguous_tensor_tensor_fast(a_t, b_t, output, dtype)

    out_shape = tuple(output.shape)
    a_stride, b_stride, out_stride = _real_layout(out_shape, a_t, b_t, output)
    if _can_use_2d_broadcast(out_shape):
        return _fallback_launch_2d_broadcast(
            a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
        )
    return _launch_generic(
        a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
    )


logger = logging.getLogger(__name__)

# Name of the active flag_gems backend device (e.g. "cuda", "musa"). The
# optimized Triton path below runs on this device; tensors on other device
# types (e.g. plain "cpu") fall through to the aten reference path. Gating on
# the hardcoded literal "cuda" wrongly excluded backends whose device type is
# not "cuda" (e.g. mthreads reports "musa") from the shared Triton path.


def _scalar_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in ((128, 1), (1024, 1), (1024, 4), (2048, 2))
    ] + [triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3)]


def _row_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in ((1024, 2), (1024, 4), (1024, 8), (2048, 1), (4096, 2))
    ] + [
        triton.Config({"BLOCK_SIZE": block}, num_warps=4, num_stages=3)
        for block in (256, 512, 1024)
    ]


def _prune_mul_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    defaults = _row_configs() if "n_cols" in args else _scalar_configs()
    default_signatures = {str(config) for config in defaults}
    # Keep YAML's expanded domain; unavailable YAML still uses default pruning.
    if any(str(config) not in default_signatures for config in configs):
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
        preferred = {(2048, 1)} if elements < 4194304 else {(4096, 2)}
    elif "n_cols" in args:
        preferred = {(1024, 8)} if elements >= 4194304 else {(1024, 2), (1024, 4)}
    elif elements < 2048:
        preferred = {(128, 1)}
    elif elements <= 32768:
        preferred = {(1024, 1), (2048, 2)}
    else:
        preferred = {(1024, 4)} if elements <= 131072 else {(2048, 2)}
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
    if dtype == "float32" and n_elements >= 2048 and n_elements <= 32768:
        offsets = tl.arange(0, BLOCK_SIZE // 4)[:, None] * 4 + tl.arange(0, 4)[None, :]
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
        dtype == "float32"
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

    # For complex32, ar*br - ai*bi subtracts cross terms, so fp16 intermediate
    # precision can cause cancellation error. Empirically, a few elements in
    # large complex32 shapes exceeded the 1e-3 relative tolerance. Compute in
    # fp32 and let tl.store cast back to the output dtype; only intermediate
    # precision changes.
    ar = tl.load(ar_ptr + ar_offsets, mask=mask, other=0.0).to(tl.float32)
    ai = tl.load(ai_ptr + ai_offsets, mask=mask, other=0.0).to(tl.float32)
    br = tl.load(br_ptr + br_offsets, mask=mask, other=0.0).to(tl.float32)
    bi = tl.load(bi_ptr + bi_offsets, mask=mask, other=0.0).to(tl.float32)

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

    # For complex32, ar*br - ai*bi subtracts cross terms, so fp16 intermediate
    # precision can cause cancellation error. Empirically, a few elements in
    # large complex32 shapes exceeded the 1e-3 relative tolerance. Compute in
    # fp32 and let tl.store cast back to the output dtype; only intermediate
    # precision changes.
    ar = tl.load(ar_ptr + ar_offsets, mask=mask, other=0.0).to(tl.float32)
    ai = tl.load(ai_ptr + ai_offsets, mask=mask, other=0.0).to(tl.float32)
    br = tl.load(br_ptr + br_offsets, mask=mask, other=0.0).to(tl.float32)
    bi = tl.load(bi_ptr + bi_offsets, mask=mask, other=0.0).to(tl.float32)

    out_r = ar * br - ai * bi
    out_i = ar * bi + ai * br
    tl.store(out_r_ptr + out_r_offsets, out_r, mask=mask)
    tl.store(out_i_ptr + out_i_offsets, out_i, mask=mask)


def _broadcast_shape(a_t, b_t):
    sa, sb = a_t.shape, b_t.shape
    if sa == sb:
        return tuple(sa)
    na, nb = len(sa), len(sb)
    n = na if na > nb else nb
    out = [0] * n
    for i in range(1, n + 1):
        da = sa[na - i] if i <= na else 1
        db = sb[nb - i] if i <= nb else 1
        if da == 1:
            out[n - i] = db
        elif db == 1 or da == db:
            out[n - i] = da
        else:
            raise RuntimeError(
                f"The size of tensor a {tuple(sa)} must match the size of "
                f"tensor b {tuple(sb)} at non-singleton dimension"
            )
    return tuple(out)


def _real_output(a_t, b_t, out=None):
    out_shape = _broadcast_shape(a_t, b_t)
    if out is None:
        if tuple(a_t.shape) == out_shape:
            return torch.empty_like(a_t)
        if tuple(b_t.shape) == out_shape:
            return torch.empty_like(b_t)
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


def _launch_contiguous_tensor_tensor(a_t, b_t, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        mul_kernel[grid](
            a_t,
            b_t,
            output,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=_is_bool_dtype(dtype),
        )
    return output


def _launch_scalar(tensor, scalar, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        mul_scalar_kernel[grid](
            tensor,
            output,
            scalar,
            n_elements,
            dtype=_dtype_name(dtype),
            IS_BOOL=_is_bool_dtype(dtype),
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
    if dtype == torch.float32 and output.is_contiguous() and n_elements < 2**31:
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
        mul_broadcast_2d_kernel[grid](
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
            IS_BOOL=_is_bool_dtype(dtype),
            DENSE_COLS=dense_cols,
        )
    return output


def mul_broadcast_func(a, b, out=None):
    if not (_is_tensor_or_number(a) and _is_tensor_or_number(b)):
        raise TypeError("mul expects tensor or scalar inputs")

    device = _select_device(a, b)
    if device.type != _DEVICE_NAME:
        if out is not None:
            return torch.ops.aten.mul.out.redispatch(_FALLBACK_KEYSET, a, b, out=out)
        return torch.ops.aten.mul.Tensor.redispatch(_FALLBACK_KEYSET, a, b)

    dtype = _result_dtype(a, b)
    if dtype != torch.float32:
        return _fallback_mul_broadcast_func(a, b, out=out)

    if (
        isinstance(a, torch.Tensor)
        and not isinstance(b, torch.Tensor)
        and not isinstance(b, complex)
    ):
        a_t = _as_tensor(a, device=device, dtype=dtype)
        output = _fallback_real_output(a_t, a_t, out=out)
        if _can_use_contiguous_scalar(a_t, output):
            return _launch_scalar(a_t, b, output, dtype)

    if (
        isinstance(b, torch.Tensor)
        and not isinstance(a, torch.Tensor)
        and not isinstance(a, complex)
    ):
        b_t = _as_tensor(b, device=device, dtype=dtype)
        output = _fallback_real_output(b_t, b_t, out=out)
        if _can_use_contiguous_scalar(b_t, output):
            return _launch_scalar(b_t, a, output, dtype)

    a_t = _as_tensor(a, device=device, dtype=dtype)
    b_t = _as_tensor(b, device=device, dtype=dtype)
    output = _real_output(a_t, b_t, out=out)

    if _can_use_contiguous_tensor_tensor(a_t, b_t, output):
        return _launch_contiguous_tensor_tensor(a_t, b_t, output, dtype)

    out_shape = tuple(output.shape)
    a_stride, b_stride, out_stride = _real_layout(out_shape, a_t, b_t, output)
    if _can_use_2d_broadcast(out_shape):
        return _launch_2d_broadcast(
            a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
        )
    return _launch_generic(
        a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
    )


def _complex_parts(value, *, device, complex_dtype):
    if isinstance(value, torch.Tensor):
        tensor = _as_tensor(value, device=device, dtype=complex_dtype)
        real_view = torch.view_as_real(tensor)
        return real_view[..., 0], real_view[..., 1]

    # FlagTree does not accept complex dtypes as Triton pointer arguments.
    # Keep Python scalar operands in their real component dtype and split the
    # value before launching the complex kernel.
    real_dtype = torch.empty((), dtype=complex_dtype).real.dtype
    if isinstance(value, complex):
        real_value, imag_value = value.real, value.imag
    else:
        real_value, imag_value = value, 0
    return (
        torch.tensor(real_value, device=device, dtype=real_dtype),
        torch.tensor(imag_value, device=device, dtype=real_dtype),
    )


def _complex_output(out_shape, *, device, dtype, out=None):
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
        if out is not None:
            return torch.ops.aten.mul.out.redispatch(_FALLBACK_KEYSET, a, b, out=out)
        return torch.ops.aten.mul.Tensor.redispatch(_FALLBACK_KEYSET, a, b)

    dtype = _result_dtype(a, b)
    ar, ai = _complex_parts(a, device=device, complex_dtype=dtype)
    br, bi = _complex_parts(b, device=device, complex_dtype=dtype)
    out_shape = tuple(
        torch.broadcast_shapes(
            tuple(ar.shape), tuple(ai.shape), tuple(br.shape), tuple(bi.shape)
        )
    )
    output = _complex_output(out_shape, device=device, dtype=dtype, out=out)
    out_view = torch.view_as_real(output)
    strides, out_r_stride, out_i_stride = _complex_layout(
        out_shape, [ar, ai, br, bi], out_view[..., 0], out_view[..., 1]
    )
    return _launch_complex_generic(
        ar, ai, br, bi, output, out_shape, strides, out_r_stride, out_i_stride, dtype
    )


def mul(A, B, *, out=None):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        result = _try_contiguous(A, B, out)
        if result is not None:
            return result
    logger.debug("GEMS_HYGON MUL")
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
    logger.debug("GEMS_HYGON MUL_")
    if not isinstance(A, torch.Tensor):
        raise TypeError("mul_ expects the first argument to be a tensor")
    dtype = _result_dtype(A, B)
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
