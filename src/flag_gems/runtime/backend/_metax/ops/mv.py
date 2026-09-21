# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX MV: select a load orientation, then launch a tuned FP32 reduction.

Dispatch selects a load orientation and optional split reduction from strides
and parallelism. Standard libtuner selects tiles within each kernel.
Tensor strides and storage offsets are consumed directly. Only compiled code
and layout choices are cached; partial results are always call-local.
"""

import copy
import logging
import os
from typing import Callable, NamedTuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.device_info import get_sm_count

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mv_metax_expand.yaml")
)

# A BLOCK_N x BLOCK_M tile is accumulated in float32 private memory, of which MetaX allows
# 4 KB per thread. A tile above _MAX_TILE_ELEMS overruns that even at the
# maximum warp count, and one above _MAX_TILE_ELEMS_PER_WARP per warp overruns
# it at the warp count it is paired with; either way the launch fails with
# mcErrorMemoryValueTooLarge instead of falling back. Every config in
# tune_configs.yaml stays under both limits, so these bounds only matter for
# candidates added later.
_MAX_TILE_ELEMS = 32768
_MAX_TILE_ELEMS_PER_WARP = 16384


def _prune_tiles(configs, named_args, **kwargs):
    # AABS may shrink a candidate in-place; keep the search space for later shapes.
    args = {**named_args, **kwargs}
    m, k = args["M"], args["K"]
    return [
        copy.deepcopy(config)
        for config in configs
        if config.kwargs["BLOCK_N"] <= max(16, triton.next_power_of_2(m))
        and config.kwargs["BLOCK_M"] <= max(128, triton.next_power_of_2(k))
        and config.kwargs["BLOCK_N"] * config.kwargs["BLOCK_M"] <= _MAX_TILE_ELEMS
        and config.kwargs["BLOCK_N"] * config.kwargs["BLOCK_M"]
        <= _MAX_TILE_ELEMS_PER_WARP * config.num_warps
    ]


_KEY = ["M", "K", "BATCH", "SAB", "SAM", "SAK", "SXB", "SXK", "SYB", "SYM", "SPLIT_K"]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv_row"),
    key=["M", "K", "SAM", "SAK", "SXK", "SYM"],
    prune_configs_by={"early_config_prune": _prune_tiles},
    rep=20,
    flagtune_op_name="mv_row",
    flagtune_expand_op_name="mv_row",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def _mv_row_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SXK: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # BLOCK_N=1 is one output per CTA; larger BLOCK_N shares x across several rows.
    rows = (tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offsets = tl.arange(0, BLOCK_M).to(tl.int64)
    acc = tl.zeros((BLOCK_N, BLOCK_M), tl.float32)
    for block in range(tl.cdiv(K, BLOCK_M)):
        ks = block * BLOCK_M + offsets
        x = tl.load(X + ks * SXK, ks < K, other=0).to(tl.float32)
        a = tl.load(
            A + rows[:, None] * SAM + ks[None, :] * SAK,
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        ).to(tl.float32)
        acc = tl.fma(x[None, :], a, acc)
    tl.store(Y + rows * SYM, tl.sum(acc, axis=1), rows < M)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv_column"),
    key=_KEY,
    prune_configs_by={"early_config_prune": _prune_tiles},
    rep=20,
    flagtune_op_name="mv_column",
    flagtune_expand_op_name="mv_column",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def _mv_column_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SXB: tl.constexpr,
    SXK: tl.constexpr,
    SYB: tl.constexpr,
    SYM: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    part = tl.program_id(1)
    batch = tl.program_id(2)
    offsets = tl.arange(0, BLOCK_M)
    # Rows are the inner dimension: contiguous column-major loads need no
    # materialized transpose, and each x element is reused across these rows.
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(part * BLOCK_M, K, SPLIT_K * BLOCK_M):
        ks = start + offsets
        a = tl.load(
            A + batch * SAB + ks[:, None] * SAK + rows[None, :] * SAM,
            (ks[:, None] < K) & (rows[None, :] < M),
            other=0,
        ).to(tl.float32)
        x = tl.load(X + batch * SXB + ks * SXK, ks < K, other=0).to(tl.float32)
        acc = tl.fma(a, x[:, None], acc)
    values = tl.sum(acc, axis=0)
    tl.store(Y + batch * SYB + part * M + rows * SYM, values, rows < M)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv_reduce"),
    key=["M", "BATCH", "SPLIT_K", "SYB", "SYM"],
    prune_configs_by={
        "early_config_prune": lambda configs, named_args, **kw: copy.deepcopy(configs)
    },
    rep=20,
    flagtune_op_name="mv_reduce",
    flagtune_expand_op_name="mv_reduce",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def _mv_reduce_kernel(
    P,
    Y,
    M: tl.constexpr,
    BATCH: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SYB: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1)
    parts = tl.arange(0, SPLIT_K)
    values = tl.load(
        P + batch * SPLIT_K * M + parts[:, None] * M + rows[None, :],
        rows[None, :] < M,
        other=0,
    )
    tl.store(Y + batch * SYB + rows * SYM, tl.sum(values, axis=0), rows < M)


class _MvPlan(NamedTuple):
    """An immutable execution choice; never owns tensors or workspace."""

    launch: Callable
    split_k: int = 1


class _MvCall(NamedTuple):
    """Tensors and metadata for one invocation."""

    a: torch.Tensor
    x: torch.Tensor
    out: torch.Tensor
    m: int
    k: int
    a_strides: tuple
    x_stride: int
    out_stride: int


def _split_count(m, k):
    """Estimate parallelism using a 128-row tile and about four CTAs per SM."""
    if k < 1024:
        return 1
    # Plain host arithmetic avoids constexpr_function wrappers on every call.
    output_tiles = (m + 127) // 128
    occupancy_target = (4 * get_sm_count() + output_tiles - 1) // output_tiles
    work_limit = (k + 63) // 64
    occupancy = 1 << (occupancy_target - 1).bit_length()
    reduction_work = 1 << (work_limit - 1).bit_length()
    return min(128, occupancy, reduction_work)


def _dispatch_mv(call):
    """Select the algorithm from metadata; never launch or benchmark candidates."""
    if call.a_strides[0] >= call.a_strides[1]:
        return _MvPlan(_launch_row)
    # Column loads expose fewer output CTAs; split long reductions for parallelism.
    return _MvPlan(_launch_column, _split_count(call.m, call.k))


def _launch_row(call, split_k):
    _mv_row_kernel[lambda cfg: (triton.cdiv(call.m, cfg["BLOCK_N"]),)](
        call.a,
        call.x,
        call.out,
        call.m,
        call.k,
        *call.a_strides,
        call.x_stride,
        call.out_stride,
    )


def _launch_column(call, split_k=-1):
    if split_k > 1:
        target = torch.empty(
            (split_k, call.m), device=call.a.device, dtype=torch.float32
        )
        target_strides = (split_k * call.m, 1)
    else:
        target, target_strides = call.out, (0, call.out_stride)
    _mv_column_kernel[lambda cfg: (triton.cdiv(call.m, cfg["BLOCK_N"]), split_k, 1)](
        call.a,
        call.x,
        target,
        call.m,
        call.k,
        1,
        0,
        *call.a_strides,
        0,
        call.x_stride,
        *target_strides,
        SPLIT_K=split_k,
    )
    if split_k > 1:
        _mv_reduce_kernel[lambda cfg: (triton.cdiv(call.m, cfg["BLOCK"]), 1)](
            target,
            call.out,
            call.m,
            1,
            split_k,
            0,
            call.out_stride,
        )


def mv(input, vec, *, out=None):
    """Dispatch the workload from tensor metadata, then launch the selected plan."""
    logger.debug("GEMS METAX MV")
    m, k = input.shape
    if out is None:
        out = torch.empty((m,), device=input.device, dtype=input.dtype)
    call = _MvCall(input, vec, out, m, k, input.stride(), vec.stride(0), out.stride(0))
    with torch_device_fn.device(input.device):
        plan = _dispatch_mv(call)
        plan.launch(call, plan.split_k)
        return call.out
