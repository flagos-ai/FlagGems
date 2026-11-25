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
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _prune_mv_configs(configs, named_args, **kwargs):
    # FP32 accumulators, converted operands, and lowering temporaries share UB.
    # Leave headroom even for a matrix with neither stride equal to one.
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_N"] <= 4096
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mv"),
    key=["M", "N"],
    strategy=["align32", "align32"],
    prune_configs_by={"early_config_prune": _prune_mv_configs},
    warmup=5,
    rep=10,
    flagtune_op_name="mv",
)
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    offset_m = tl.arange(0, BLOCK_M)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for m in range(0, M, BLOCK_M):
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


@libentry()
@triton.jit
def _mv_column_major(
    A,
    B,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_AK: tl.constexpr,
    STRIDE_BK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    ks = split * CHUNK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    for offset in range(0, CHUNK_K, BLOCK_K):
        k = ks + offset
        a = tl.load(
            A + k[:, None].to(tl.int64) * STRIDE_AK + rows[None, :],
            (k[:, None] < K) & (rows[None, :] < N),
            other=0,
        ).to(tl.float32)
        b = tl.load(B + k.to(tl.int64) * STRIDE_BK, k < K, other=0).to(tl.float32)
        acc += a * b[:, None]
    tl.store(C + split * N + rows, tl.sum(acc, axis=0), rows < N)


@libentry()
@triton.jit
def _mv_merge(
    P,
    C,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    splits = tl.arange(0, BLOCK_SPLITS)
    parts = tl.load(
        P + splits[:, None] * N + rows[None, :],
        (splits[:, None] < SPLITS) & (rows[None, :] < N),
        other=0,
    )
    tl.store(C + rows, tl.sum(parts, axis=0), rows < N)


def _launch_column_major(inp, vec, out, n, k):
    # Keep the physically contiguous row axis innermost. Narrow matrices lack
    # row tiles; splitting K supplies more work without growing the UB tile.
    block_n = 16 if n <= 64 else 64
    block_k = 128 if block_n == 16 else 64
    row_tiles = triton.cdiv(n, block_n)
    target_splits = triton.cdiv(32, row_tiles)
    chunk_k = triton.cdiv(k, target_splits * block_k) * block_k
    splits = triton.cdiv(k, chunk_k)
    partial = (
        torch.empty((splits, n), device=out.device, dtype=torch.float32)
        if splits > 1
        else out
    )
    _mv_column_major[(row_tiles, splits)](
        inp,
        vec,
        partial,
        n,
        k,
        inp.stride(1),
        vec.stride(0),
        block_n,
        block_k,
        chunk_k,
    )
    if splits > 1:
        _mv_merge[(triton.cdiv(n, 128),)](
            partial, out, n, splits, 128, triton.next_power_of_2(splits)
        )


def mv(inp, vec):
    logger.debug("GEMS_ASCEND MV")
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    if N == 0:
        return out
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(inp.device):
        if M > 0 and inp.stride(0) == 1 and inp.stride(1) != 1:
            _launch_column_major(inp, vec, out, N, M)
            return out
        mv_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
        )
    return out
