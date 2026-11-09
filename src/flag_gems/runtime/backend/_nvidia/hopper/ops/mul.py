# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hopper contiguous, scalar and dense broadcast multiplication for FP32 and BF16."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.mul import (
    _contiguous_configs,
    _dtype_name,
    _mul,
    _mul_contiguous_tile,
    mul_broadcast_get_configs,
    mul_get_configs,
    mul_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.device_info import get_sm_count


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
        and config.num_stages == 1
    ]
    return selected or configs


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
def mul_contiguous_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BOOL: tl.constexpr,
    EVEN_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr = 0,
):
    # The host launcher already restricts this path to n_elements < 2**31.
    if dtype == "float16" or dtype == "float32" or dtype == "bfloat16":
        # NUM_SMS is nonzero only for the validated Hopper BF16 launch policy.
        if NUM_SMS > 0 and tl.cdiv(n_elements, BLOCK_SIZE) > NUM_SMS * 8:
            for block in range(
                tl.program_id(0), tl.cdiv(n_elements, BLOCK_SIZE), tl.num_programs(0)
            ):
                _mul_contiguous_tile(
                    x_ptr, y_ptr, output_ptr, n_elements, block, BLOCK_SIZE, EVEN_SIZE
                )
        else:
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
            num_sms = 0
            if dtype == torch.bfloat16 and 1048576 <= n_elements <= 4194304:
                num_sms = get_sm_count()

            def contiguous_grid(meta):
                blocks = triton.cdiv(n_elements, meta["BLOCK_SIZE"])
                return (min(blocks, num_sms * 8) if num_sms else blocks,)

            mul_contiguous_kernel[contiguous_grid](
                a_t,
                b_t,
                output,
                n_elements,
                dtype=_dtype_name(dtype),
                IS_BOOL=dtype is torch.bool,
                NUM_SMS=num_sms,
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


def _launch_scalar(tensor, scalar, output, dtype):
    n_elements = output.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(output.device):
        kernel = mul_scalar_kernel
        launch = {}
        if torch._C._is_alias_of(output, tensor):
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
    rows, cols = out_shape
    if dtype in (torch.float32, torch.bfloat16) and output.is_contiguous():
        if b_stride == (1, 0) and a_stride in ((cols, 1), (0, 1)):
            a_t, b_t = b_t, a_t
            a_stride, b_stride = b_stride, a_stride
    tiled = (
        dtype in (torch.float32, torch.bfloat16)
        and n_elements < 2**31
        and a_stride == (1, 0)
        and b_stride in ((cols, 1), (0, 1))
        and out_stride == (cols, 1)
        and not (dtype == torch.float32 and n_elements <= 65536)
    )

    def grid(meta):
        block = meta["BLOCK_SIZE"]
        if not tiled or (
            b_stride[0] == 0 and cols & (cols - 1) and n_elements <= 1048576
        ):
            return (triton.cdiv(n_elements, block),)
        block_n = min(block, triton.next_power_of_2(cols))
        block_m = max(1, block // block_n)
        return (triton.cdiv(rows, block_m) * triton.cdiv(cols, block_n),)

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
            **launch,
        )
    return output


def _scalar_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in (
            (256, 4),
            (512, 4),
            (512, 2),
            (1024, 4),
            (2048, 4),
            (1024, 8),
            (4096, 8),
        )
    ] + mul_get_configs()


def _broadcast_configs():
    return [
        triton.Config({"BLOCK_SIZE": block}, num_warps=warps, num_stages=1)
        for block, warps in (
            (256, 4),
            (512, 4),
            (1024, 4),
            (1024, 1),
            (1024, 2),
            (2048, 4),
            (4096, 4),
            (4096, 2),
            (8192, 4),
            (1024, 8),
            (2048, 8),
        )
    ] + mul_broadcast_get_configs()


def _prune_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    broadcast = "n_cols" in args
    defaults = _broadcast_configs() if broadcast else _scalar_configs()
    default_signatures = {str(config) for config in defaults}
    # Preserve the complete YAML domain when Expanded is enabled.
    if any(str(config) not in default_signatures for config in configs):
        return configs
    if args.get("dtype") not in ("float32", "bfloat16") or (
        broadcast
        and args.get("dtype") == "float32"
        and args.get("n_elements", 0) <= 65536
    ):
        blocks = (256, 512, 1024) if broadcast else (1024,)
        return [
            c
            for c in configs
            if c.num_warps == 4
            and c.num_stages == 3
            and c.kwargs["BLOCK_SIZE"] in blocks
        ]
    elements = args.get("n_elements", 0)
    if not broadcast and args.get("dtype") == "float32" and 32768 <= elements <= 262144:
        preferred = {(1024, 8)}
    elif elements <= 65536:
        preferred = {(1024, 4), (1024, 8)}
        if not broadcast:
            preferred.add((512, 4))
    elif not broadcast:
        preferred = {(512, 2), (1024, 4), (1024, 8), (2048, 4), (4096, 8)}
    elif args.get("b_s0") == 0:
        preferred = (
            {(256, 4), (512, 4), (1024, 8)}
            if elements <= 1048576
            else {(4096, 4), (8192, 4)}
        )
    elif elements <= 4194304:
        preferred = {(1024, 1), (1024, 2), (1024, 4), (1024, 8), (2048, 4)}
    else:
        preferred = {
            (1024, 1),
            (1024, 2),
            (1024, 8),
            (2048, 4),
            (2048, 8),
            (4096, 2),
            (4096, 4),
            (8192, 4),
        }
    selected = [
        config
        for config in configs
        if config.num_stages == 1
        and (config.kwargs["BLOCK_SIZE"], config.num_warps) in preferred
    ]
    return selected or configs


@libentry()
@libtuner(
    configs=_scalar_configs(),
    key=["n_elements", "dtype"],
    strategy=["align32", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _prune_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul",
    flagtune_op_id="flaggems/mul",
    flagtune_variant="scalar",
    policy="flagtune",
)
@triton.heuristics(
    {"EVEN_SIZE": lambda args: args["n_elements"] % args["BLOCK_SIZE"] == 0}
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
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if EVEN_SIZE:
        x = tl.load(x_ptr + offsets)
        if dtype == "float32" or dtype == "bfloat16":
            x = x.to(tl.float32)
        value = x & scalar if IS_BOOL else x * scalar
        tl.store(output_ptr + offsets, value)
    else:
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask, other=0)
        if dtype == "float32" or dtype == "bfloat16":
            x = x.to(tl.float32)
        value = x & scalar if IS_BOOL else x * scalar
        tl.store(output_ptr + offsets, value, mask)


@libentry()
@libtuner(
    configs=_broadcast_configs(),
    key=["n_elements", "n_cols", "dtype"],
    strategy=["align32", "default", "default"],
    warmup=5,
    rep=20,
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": _prune_configs},
    flagtune_op_name="mul",
    flagtune_expand_op_name="mul_broadcast_2d",
    flagtune_op_id="flaggems/mul",
    flagtune_variant="broadcast_2d",
    policy="flagtune",
)
@triton.heuristics(
    {
        "TILED": lambda args: args["n_elements"] < 2**31
        and not (args["dtype"] == "float32" and args["n_elements"] <= 65536),
        "SMALL_OUTER": lambda args: args["n_elements"] <= 1048576,
    }
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
    TILED: tl.constexpr,
    SMALL_OUTER: tl.constexpr,
):
    if (
        (dtype == "float32" or dtype == "bfloat16")
        and TILED
        and a_s0 == 1
        and a_s1 == 0
        and b_s1 == 1
        and (b_s0 == n_cols or b_s0 == 0)
        and out_s0 == n_cols
        and out_s1 == 1
    ):
        # Small non-power-of-two outer products avoid padding each row to a tile.
        if b_s0 == 0 and n_cols & (n_cols - 1) and SMALL_OUTER:
            offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            a = tl.load(a_ptr + offsets // n_cols, mask, other=0).to(tl.float32)
            b = tl.load(b_ptr + offsets % n_cols, mask, other=0).to(tl.float32)
            tl.store(out_ptr + offsets, a * b, mask)
        else:
            block_n: tl.constexpr = min(BLOCK_SIZE, triton.next_power_of_2(n_cols))
            block_m: tl.constexpr = max(1, BLOCK_SIZE // block_n)
            col_tiles: tl.constexpr = tl.cdiv(n_cols, block_n)
            pid = tl.program_id(0)
            rows = (pid // col_tiles) * block_m + tl.arange(0, block_m)
            cols = (pid % col_tiles) * block_n + tl.arange(0, block_n)
            mask = (rows[:, None] < n_elements // n_cols) & (cols[None, :] < n_cols)
            a = tl.load(a_ptr + rows, rows < n_elements // n_cols, other=0).to(
                tl.float32
            )
            if b_s0 == 0:
                b = tl.load(b_ptr + cols, cols < n_cols, other=0).to(tl.float32)
                value = a[:, None] * b[None, :]
            else:
                b = tl.load(
                    b_ptr + rows[:, None] * n_cols + cols[None, :], mask, other=0
                ).to(tl.float32)
                value = a[:, None] * b
            tl.store(out_ptr + rows[:, None] * n_cols + cols[None, :], value, mask)
    else:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        row = offsets // n_cols
        col = offsets % n_cols
        a = tl.load(a_ptr + row * a_s0 + col * a_s1, mask, other=0)
        b = tl.load(b_ptr + row * b_s0 + col * b_s1, mask, other=0)
        value = a & b if IS_BOOL else a * b
        tl.store(out_ptr + row * out_s0 + col * out_s1, value, mask)


# Reuse common validation and fallbacks with this backend's launch policies.
_REAL_LAUNCHERS = (
    _launch_contiguous_tensor_tensor,
    _launch_scalar,
    _launch_2d_broadcast,
)


def mul(A, B, *, out=None):
    return _mul(A, B, out=out, launchers=_REAL_LAUNCHERS)
