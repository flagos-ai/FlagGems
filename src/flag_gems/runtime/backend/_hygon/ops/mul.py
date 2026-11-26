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

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops.mul import _as_tensor
from flag_gems.ops.mul import _complex_output as _output
from flag_gems.ops.mul import (
    _contiguous_configs,
    _dtype_name,
    _mul,
    _mul_,
    _mul_contiguous_tile,
    _prune_contiguous_configs,
    mul_broadcast_get_configs,
)
from flag_gems.ops.mul import mul_complex_broadcast_func as _mul_complex_broadcast_func
from flag_gems.ops.mul import mul_kernel
from flag_gems.runtime import flagtune_enabled, torch_device_fn
from flag_gems.utils import libentry, libtuner


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
@triton.heuristics(
    {"EVEN_SIZE": lambda args: args["n_elements"] % args["BLOCK_SIZE"] == 0}
)
@triton.jit(do_not_specialize=["x_ptr", "y_ptr", "output_ptr"])
def mul_contiguous_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    EVEN_SIZE: tl.constexpr,
):
    # The host launcher already restricts this path to n_elements < 2**31.
    if dtype == "float16" or dtype == "float32" or dtype == "bfloat16":
        _mul_contiguous_tile(
            x_ptr,
            y_ptr,
            output_ptr,
            n_elements,
            tl.program_id(0),
            BLOCK_SIZE,
            EVEN_SIZE,
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
@triton.heuristics(
    {"EVEN_SIZE": lambda args: args["n_elements"] % args["BLOCK_SIZE"] == 0}
)
@triton.jit
def mul_contiguous_bf16_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    EVEN_SIZE: tl.constexpr,
):
    # Pointer specialization lets HCU vectorize BF16 memory accesses. Keep the
    # original entry for other dtypes, whose small shapes prefer its layout.
    _mul_contiguous_tile(
        x_ptr, y_ptr, output_ptr, n_elements, tl.program_id(0), BLOCK_SIZE, EVEN_SIZE
    )


_hygon_contiguous_default = libentry()(mul_contiguous_kernel.fn.fn)
_hygon_bf16_contiguous_default = libentry()(mul_contiguous_bf16_kernel.fn.fn)


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


def _launch_contiguous_tensor_tensor(a_t, b_t, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    use_bf16 = dtype == torch.bfloat16 and n_elements >= 65536
    kernel = mul_contiguous_bf16_kernel if use_bf16 else mul_contiguous_kernel
    default_kernel = (
        _hygon_bf16_contiguous_default if use_bf16 else _hygon_contiguous_default
    )
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    guard = (
        nullcontext()
        if torch_device_fn.current_device() == output.device.index
        else torch_device_fn.device(output.device)
    )
    with guard:
        if (
            n_elements < 2**31
            and not torch._C._is_alias_of(output, a_t)
            and not torch._C._is_alias_of(output, b_t)
        ):
            if (
                dtype in (torch.float32, torch.bfloat16)
                and n_elements <= 4194304
                and kernel.fn._run_mode == "normal"
                and not flagtune_enabled("mul")
            ):
                block, warps = _hygon_contiguous_config(n_elements, dtype)
                default_kernel[grid](
                    a_t,
                    b_t,
                    output,
                    n_elements,
                    dtype=_dtype_name(dtype),
                    IS_BOOL=dtype is torch.bool,
                    BLOCK_SIZE=block,
                    num_warps=warps,
                    num_stages=1,
                )
                return output
            kernel[grid](
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
    # BF16 shares the kernels without inheriting FP32-only preferences.
    if args.get("dtype") == "bfloat16" and "n_cols" in args:
        return configs
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
    n_elements,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EVEN_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN_SIZE:
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
    n_elements,
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
@triton.heuristics(
    {
        "EVEN_SIZE": lambda args: args["n_elements"] % args["BLOCK_SIZE"] == 0,
        "SINGLE_BLOCK": lambda args: args["n_elements"] <= args["BLOCK_SIZE"],
        "PACKED": lambda args: args["dtype"] == "float32"
        and 2048 <= args["n_elements"] <= 32768,
        "PAIR_TILES": lambda args: args["dtype"] == "bfloat16"
        and args["n_elements"] >= 262144
        and args["n_elements"] % (2 * args["BLOCK_SIZE"]) == 0,
    }
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
    EVEN_SIZE: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    PACKED: tl.constexpr,
    PAIR_TILES: tl.constexpr,
):
    if PAIR_TILES:
        # Merge two full tiles so the compiler can emit wider BF16 memory accesses.
        offsets = tl.program_id(0) * (2 * BLOCK_SIZE) + tl.arange(0, 2 * BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        tl.store(output_ptr + offsets, x * scalar)
        return

    if PACKED:
        offsets = tl.arange(0, BLOCK_SIZE // 4)[:, None] * 4 + tl.arange(0, 4)[None, :]
    else:
        offsets = tl.arange(0, BLOCK_SIZE)
    if not SINGLE_BLOCK:
        offsets += tl.program_id(0) * BLOCK_SIZE
    # Exact tiles need no per-element bounds checks. All launchers use ceildiv
    # grids, so only the final partial tile requires a mask.
    if EVEN_SIZE:
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
@triton.heuristics(
    {"EVEN_SIZE": lambda args: args["n_elements"] % args["BLOCK_SIZE"] == 0}
)
@triton.jit
def mul_broadcast_2d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
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
    EVEN_SIZE: tl.constexpr,
    DENSE_COLS: tl.constexpr = 0,
):
    if DENSE_COLS:
        _mul_dense_row(
            a_ptr, b_ptr, out_ptr, n_elements, DENSE_COLS, BLOCK_SIZE, EVEN_SIZE
        )
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
    return _output(out_shape, device=a_t.device, dtype=a_t.dtype, out=out)


def _launch_scalar(tensor, scalar, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output

    def grid(meta):
        block = meta["BLOCK_SIZE"]
        if (
            dtype == torch.bfloat16
            and n_elements >= 262144
            and n_elements % (2 * block) == 0
        ):
            block *= 2
        return (triton.cdiv(n_elements, block),)

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


def mul_complex_broadcast_func(a, b, out=None):
    return _mul_complex_broadcast_func(
        a,
        b,
        out=out,
        parts=_complex_parts,
        kernels=(
            mul_complex_generic_nd_kernel,
            mul_complex_generic_nd_runtime_meta_kernel,
        ),
    )


# Reuse common validation and fallbacks with this backend's launch policies.
_REAL_LAUNCHERS = (
    _launch_contiguous_tensor_tensor,
    _launch_scalar,
    _launch_2d_broadcast,
)


def mul(A, B, *, out=None):
    return _mul(
        A,
        B,
        out=out,
        launchers=_REAL_LAUNCHERS,
        real_output=_real_output,
        complex_mul=mul_complex_broadcast_func,
    )


def mul_(A, B):
    return _mul_(
        A,
        B,
        launchers=_REAL_LAUNCHERS,
        real_output=_real_output,
        complex_mul=mul_complex_broadcast_func,
    )
