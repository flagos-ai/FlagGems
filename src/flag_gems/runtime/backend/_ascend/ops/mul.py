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
import math
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device as runtime_device
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.libentry import LibTuner
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

_DEVICE_NAME = runtime_device.name

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _prune_mul_flat_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    if _DEVICE_NAME != "npu":
        return configs
    n = args["n_elements"]
    max_block = max(128, triton.next_power_of_2(max(1, n)))
    usable = [c for c in configs if c.kwargs["BLOCK_SIZE"] <= max_block]
    if not usable:
        usable = [min(configs, key=lambda c: c.kwargs["BLOCK_SIZE"])]
    counts = {id(c): triton.cdiv(n, c.kwargs["BLOCK_SIZE"]) for c in usable}
    limit = max(1, 2 * min(counts.values()))
    return [c for c in usable if counts[id(c)] <= limit]


def mul_get_configs():
    return [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ]


def _prune_mul_full_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    n = args["n_elements"]
    max_block = 16384 if args["dtype"] == "float16" else 8192
    if n < 8192:
        max_block = min(max_block, 2048)
    usable = [
        c
        for c in configs
        if c.kwargs["BLOCK_SIZE"] <= max_block and n % c.kwargs["BLOCK_SIZE"] == 0
    ]
    if not usable:
        raise RuntimeError("No aligned MUL configuration after pruning")
    largest = max(c.kwargs["BLOCK_SIZE"] for c in usable)
    return [c for c in usable if c.kwargs["BLOCK_SIZE"] == largest]


def mul_broadcast_get_configs():
    return [
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
    ]


@libentry()
@libtuner(
    configs=mul_get_configs(),
    prune_configs_by={"early_config_prune": _prune_mul_flat_configs},
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
    configs=runtime.get_tuned_config("mul_flat"),
    key=["n_elements", "dtype"],
    strategy=["default", "default"],
    prune_configs_by={"early_config_prune": _prune_mul_full_configs},
    warmup=5,
    rep=5,
)
@triton.jit
def mul_contiguous_full_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    out = x & y if IS_BOOL else x * y
    tl.store(output_ptr + offsets, out)


@libentry()
@triton.jit
def mul_contiguous_chunked_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    IS_BOOL: tl.constexpr,
):
    pid = tl.program_id(0)
    for block_offset in tl.static_range(0, BLOCKS_PER_PROGRAM):
        block_id = pid * BLOCKS_PER_PROGRAM + block_offset
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        out = x & y if IS_BOOL else x * y
        tl.store(output_ptr + offsets, out)


@libentry()
@libtuner(
    configs=mul_get_configs(),
    prune_configs_by={"early_config_prune": _prune_mul_flat_configs},
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=5,
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


def _can_use_contiguous_scalar(tensor, out):
    return tensor.is_contiguous() and out.is_contiguous() and not tensor.is_complex()


def _real_layout(out_shape, a_t, b_t, out):
    a_stride = _broadcasted_stride(tuple(a_t.shape), tuple(a_t.stride()), out_shape)
    b_stride = _broadcasted_stride(tuple(b_t.shape), tuple(b_t.stride()), out_shape)
    out_stride = tuple(out.stride()) if out_shape else ()
    return a_stride, b_stride, out_stride


def _can_use_2d_broadcast(out_shape):
    return len(out_shape) == 2


def _launch_contiguous_tensor_tensor(a_t, b_t, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    if dtype in (torch.float16, torch.float32, torch.bfloat16) and n_elements % 64 == 0:
        max_programs = 65535

        chunk_block_size = 8192
        chunk_blocks = triton.cdiv(n_elements, chunk_block_size)
        if chunk_blocks > max_programs:
            blocks_per_program = triton.cdiv(chunk_blocks, max_programs)
            grid = (triton.cdiv(chunk_blocks, blocks_per_program),)
            with torch_device_fn.device(output.device):
                mul_contiguous_chunked_kernel[grid](
                    a_t,
                    b_t,
                    output,
                    BLOCK_SIZE=chunk_block_size,
                    BLOCKS_PER_PROGRAM=blocks_per_program,
                    IS_BOOL=_is_bool_dtype(dtype),
                )
            return output

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_device_fn.device(output.device):
            mul_contiguous_full_kernel[grid](
                a_t,
                b_t,
                output,
                n_elements,
                dtype=_dtype_name(dtype),
                IS_BOOL=_is_bool_dtype(dtype),
            )
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


def _prune_mul_tiled_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n = args["M"], args["N"]
    max_m = triton.next_power_of_2(m)
    max_n = max(32, triton.next_power_of_2(n))
    dense_operand = (args["a_s0"] != 0 and args["a_s1"] != 0) or (
        args["b_s0"] != 0 and args["b_s1"] != 0
    )
    max_elements = 8192 if args["dtype"] == "bfloat16" and dense_operand else 16384
    pruned = [
        c
        for c in configs
        if c.kwargs["BLOCK_M"] <= max_m
        and c.kwargs["BLOCK_N"] <= max_n
        and c.kwargs["BLOCK_M"] * c.kwargs["BLOCK_N"] <= max_elements
    ]
    if not pruned:
        raise RuntimeError("No valid MUL tiled configurations after pruning")
    widest = max(c.kwargs["BLOCK_N"] for c in pruned)
    wide = [c for c in pruned if c.kwargs["BLOCK_N"] >= max(32, widest // 2)]
    counts = {
        id(c): triton.cdiv(m, c.kwargs["BLOCK_M"]) * triton.cdiv(n, c.kwargs["BLOCK_N"])
        for c in wide
    }
    limit = max(1, 2 * min(counts.values()))
    return [c for c in wide if counts[id(c)] <= limit]


class _MulTiledTuner(LibTuner):
    @staticmethod
    def policy(bench_fn, configs, args, kwargs):
        timings = {}
        for config in configs:
            try:
                measured = bench_fn(config)
            except Exception as exc:
                if type(exc).__name__ != "MLIRCompilationError" or not type(
                    exc
                ).__module__.startswith("triton."):
                    raise
                logger.warning("MUL tiled compile rejected %s: %s", config, exc)
                measured = [float("inf")] * 3
            timings[config] = measured
        valid = {c: t for c, t in timings.items() if all(math.isfinite(v) for v in t)}
        if not valid:
            raise RuntimeError(
                "All MUL tiled configurations failed; inspect compile logs"
            )
        return min(valid, key=valid.get), timings


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mul_broadcast_tiled"),
    key=["M", "N", "dtype", "a_s0", "a_s1", "b_s0", "b_s1"],
    strategy=["default"] * 7,
    prune_configs_by={"early_config_prune": _prune_mul_tiled_configs},
    warmup=5,
    rep=5,
    policy=_MulTiledTuner,
)
@triton.jit
def mul_broadcast_tiled_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    a_s0: tl.constexpr,
    a_s1: tl.constexpr,
    b_s0: tl.constexpr,
    b_s1: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    col_tiles = tl.cdiv(N, BLOCK_N)
    pid = tl.program_id(0)
    rows = (pid // col_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = (pid % col_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = (rows[:, None] < M) & (cols[None, :] < N)

    if a_s0 == 0 and a_s1 == 0:
        a_scalar = tl.load(a_ptr)
        a_value = tl.broadcast_to(a_scalar, (BLOCK_M, BLOCK_N))
    elif a_s1 == 0:
        a_rows = tl.load(a_ptr + rows * a_s0, mask=rows < M, other=0)
        a_value = tl.broadcast_to(a_rows[:, None], (BLOCK_M, BLOCK_N))
    elif a_s0 == 0:
        a_cols = tl.load(a_ptr + cols * a_s1, mask=cols < N, other=0)
        a_value = tl.broadcast_to(a_cols[None, :], (BLOCK_M, BLOCK_N))
    else:
        a_value = tl.load(
            a_ptr + rows[:, None] * a_s0 + cols[None, :] * a_s1, mask=valid, other=0
        )

    if b_s0 == 0 and b_s1 == 0:
        b_scalar = tl.load(b_ptr)
        b_value = tl.broadcast_to(b_scalar, (BLOCK_M, BLOCK_N))
    elif b_s1 == 0:
        b_rows = tl.load(b_ptr + rows * b_s0, mask=rows < M, other=0)
        b_value = tl.broadcast_to(b_rows[:, None], (BLOCK_M, BLOCK_N))
    elif b_s0 == 0:
        b_cols = tl.load(b_ptr + cols * b_s1, mask=cols < N, other=0)
        b_value = tl.broadcast_to(b_cols[None, :], (BLOCK_M, BLOCK_N))
    else:
        b_value = tl.load(
            b_ptr + rows[:, None] * b_s0 + cols[None, :] * b_s1, mask=valid, other=0
        )

    result = a_value * b_value
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], result, mask=valid)


def _launch_2d_broadcast(
    a_t, b_t, output, out_shape, a_stride, b_stride, out_stride, dtype
):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    if (
        output.device.type == "npu"
        and dtype in (torch.float16, torch.float32, torch.bfloat16)
        and a_t.dtype == b_t.dtype == output.dtype == dtype
        and output.is_contiguous()
        and all(s >= 0 for s in (*a_stride, *b_stride))
        and output.untyped_storage().data_ptr() != a_t.untyped_storage().data_ptr()
        and output.untyped_storage().data_ptr() != b_t.untyped_storage().data_ptr()
    ):
        grid = lambda meta: (
            triton.cdiv(out_shape[0], meta["BLOCK_M"])
            * triton.cdiv(out_shape[1], meta["BLOCK_N"]),
        )
        with torch_device_fn.device(output.device):
            mul_broadcast_tiled_kernel[grid](
                a_t,
                b_t,
                output,
                out_shape[0],
                out_shape[1],
                a_s0=a_stride[0],
                a_s1=a_stride[1],
                b_s0=b_stride[0],
                b_s1=b_stride[1],
                dtype=_dtype_name(dtype),
            )
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        mul_broadcast_2d_kernel[grid](
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


def mul_broadcast_func(a, b, out=None):
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
        output = _real_output(a_t, a_t.new_empty(()), out=out)
        if _can_use_contiguous_scalar(a_t, output):
            return _launch_scalar(a_t, b, output, dtype)

    if (
        isinstance(b, torch.Tensor)
        and not isinstance(a, torch.Tensor)
        and not isinstance(a, complex)
    ):
        b_t = _as_tensor(b, device=device, dtype=dtype)
        output = _real_output(b_t.new_empty(()), b_t, out=out)
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
    tensor = _as_tensor(value, device=device, dtype=complex_dtype)
    real_view = torch.view_as_real(tensor)
    return real_view[..., 0], real_view[..., 1]


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
    logger.debug("GEMS_ASCEND MUL")
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
    logger.debug("GEMS_ASCEND MUL_")
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
