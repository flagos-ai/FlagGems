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


"""Shared split-K path for GEMMs with too few output tiles to fill the device."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime.flagtune import _flagtune_state_key
from flag_gems.utils import libentry, libtuner

from ..gemm_utils import (
    HAS_FULL_LAUNCH_ARGUMENTS,
    GemmChoiceCache,
    bind_gemm_launch,
    gemm_storage_is_wide,
)


@triton.jit
def _partials(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AB: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BATCH: tl.constexpr,
    SPLITS: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    split = tl.program_id(2)
    gn = tl.cdiv(N, TN)
    rm = tile // gn * TM + tl.arange(0, TM)
    rn = tile % gn * TN + tl.arange(0, TN)
    rk = tl.arange(0, TK)
    segment: tl.constexpr = triton.cdiv(K, SPLITS * TK) * TK
    start = split * segment
    acc = tl.zeros((TM, TN), tl.float32)
    for offset in tl.range(0, segment, TK, num_stages=2):
        kk = start + offset + rk
        a = tl.load(
            A
            + batch.to(tl.int64) * AB
            + rm[:, None].to(tl.int64) * AM
            + kk[None, :].to(tl.int64) * AK,
            (rm[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            B
            + batch.to(tl.int64) * BB
            + kk[:, None].to(tl.int64) * BK
            + rn[None, :].to(tl.int64) * BN,
            (kk[:, None] < K) & (rn[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    offsets = (
        (split * BATCH + batch).to(tl.int64) * M * N
        + rm[:, None].to(tl.int64) * N
        + rn[None, :]
    )
    tl.store(P + offsets, acc, (rm[:, None] < M) & (rn[None, :] < N))


@libentry(cache_binding=True)
@libtuner(
    configs=runtime.get_tuned_config("bmm_splitk"),
    key=["BATCH", "M", "N", "K", "AB", "AM", "AK", "BB", "BK", "BN", "SPLITS"],
    use_cuda_graph=True,
    warmup=5,
    rep=20,
)
@triton.jit
def _small_partials(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AB: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BATCH: tl.constexpr,
    SPLITS: tl.constexpr,
    TM: tl.constexpr,
    TN: tl.constexpr,
    TK: tl.constexpr,
):
    _partials(A, B, P, M, N, K, AB, AM, AK, BB, BK, BN, BATCH, SPLITS, TM, TN, TK)


@libentry(cache_binding=True)
@triton.jit
def _finish_splitk(
    P,
    O,
    Bias,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    OB: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    CB: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    size: tl.constexpr = BATCH * M * N
    split = tl.arange(0, SPLITS)
    values = tl.load(
        P + split[:, None].to(tl.int64) * size + idx[None, :],
        idx[None, :] < size,
        other=0,
    )
    result = tl.sum(values, axis=0)
    batch = idx // (M * N)
    row = idx // N % M
    col = idx % N
    result *= ALPHA
    if HAS_BIAS and BETA != 0:
        result += BETA * tl.load(
            Bias
            + batch.to(tl.int64) * CB
            + row.to(tl.int64) * CM
            + col.to(tl.int64) * CN,
            idx < size,
            other=0,
        ).to(tl.float32)
    tl.store(
        O + batch.to(tl.int64) * OB + row.to(tl.int64) * OM + col.to(tl.int64) * ON,
        result,
        idx < size,
    )


def _reduction_launch(
    partials, out, splits, bias=None, alpha=1, beta=0, *, bias_strides=None
):
    unbatched = out.ndim == 2
    batch, m, n = (1, *out.shape) if unbatched else out.shape
    out_strides = (0, *out.stride()) if unbatched else out.stride()
    if bias_strides is None:
        bias_strides = bias.stride() if bias is not None else (0, 0, 0)
    return (triton.cdiv(batch * m * n, 256), 1, 1), (
        partials,
        out,
        bias if bias is not None else out,
        m,
        n,
        batch,
        *out_strides,
        splits,
        256,
        *bias_strides,
        alpha,
        beta,
        bias is not None,
    )


def _launch_reduction(partials, out, splits):
    grid, args = _reduction_launch(partials, out, splits)
    _finish_splitk[grid](*args, num_warps=4, num_stages=1)


def launch_splitk(a, b, out, config):
    batch, m, k = a.shape
    n = b.shape[2]
    splits, tm, tn, tk = config
    partials = torch.empty((splits, batch, m, n), dtype=torch.float32, device=a.device)
    _partials[(triton.cdiv(m, tm) * triton.cdiv(n, tn), batch, splits)](
        a,
        b,
        partials,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        batch,
        splits,
        tm,
        tn,
        tk,
        num_warps=4,
        num_stages=2,
        allow_flush_denorm=True,
    )
    _launch_reduction(partials, out, splits)


@lru_cache(maxsize=512)
def _small_split_count(batch, m, n, k):
    if min(batch, m, n, k) <= 0 or k < 1024 or min(m, n) > 128 or max(m, n) > 4096:
        return 0
    tiles = batch * triton.cdiv(m, 32) * triton.cdiv(n, 64)
    if tiles >= 128:
        return 0
    splits = min(32, triton.next_power_of_2(triton.cdiv(128, tiles)))
    splits = min(splits, triton.next_power_of_2(k) // 256)
    return splits if splits >= 2 else 0


# Compiled launchers retain static geometry only. The workspace, output, and
# operand addresses always come from the current invocation.
_small_launches = GemmChoiceCache(maxsize=512)


def _small_launch_key(a, b, out, bias, geometry, alpha, beta):
    if (
        not HAS_FULL_LAUNCH_ARGUMENTS
        or _small_partials.fn._run_mode != "normal"
        or triton.knobs.autotuning.adjust_block_size
    ):
        return None
    if type(alpha) not in (int, float, bool) or type(beta) not in (int, float, bool):
        return None
    tensors = (a, b, out) if bias is None else (a, b, out, bias)
    if any(type(t) is not torch.Tensor or t.device != a.device for t in tensors):
        return None
    return (
        a.device,
        geometry,
        tuple(
            (t.dtype, t.data_ptr() % 16 == 0, gemm_storage_is_wide(t)) for t in tensors
        ),
        type(alpha),
        alpha,
        type(beta),
        beta,
        _flagtune_state_key(),
        _small_partials.fn._benchmark_protocol,
        triton.knobs.runtime.override_arch,
        triton.knobs.hcu.use_buffer_ops,
    )


def launch_small_splitk(a, b, out, bias=None, alpha=1, beta=0, *, bias_strides=None):
    """Return whether a long reduction was split across additional workgroups.

    Each call allocates fresh FP32 partials. The second kernel reduces them and
    applies the optional bias once, before casting to the output dtype.
    Rank-two inputs share the batch-one kernel. Supply broadcast bias strides
    when the bias does not already have three dimensions.
    """
    unbatched = a.ndim == 2
    batch, m, k = (1, *a.shape) if unbatched else a.shape
    n = b.shape[-1]
    if a.dtype not in (torch.float16, torch.bfloat16):
        return False
    splits = _small_split_count(batch, m, n, k)
    if not splits:
        return False
    partials = torch.empty((splits, batch, m, n), dtype=torch.float32, device=a.device)
    a_strides = (0, *a.stride()) if unbatched else a.stride()
    b_strides = (0, *b.stride()) if unbatched else b.stride()
    post_grid, post_args = _reduction_launch(
        partials, out, splits, bias, alpha, beta, bias_strides=bias_strides
    )
    geometry = (k, a_strides, b_strides, post_args[3:])
    key = _small_launch_key(a, b, out, bias, geometry, alpha, beta)
    plan = _small_launches.get(key) if key is not None else None
    if plan is not None:
        launch_partial, partial_constants, launch_post, post_constants = plan
        launch_partial(a, b, partials, *partial_constants)
        launch_post(partials, out, post_args[2], *post_constants)
        return True

    def partial_grid(meta):
        return (
            triton.cdiv(m, meta["TM"]) * triton.cdiv(n, meta["TN"]),
            batch,
            splits,
        )

    partial_constants = (m, n, k, *a_strides, *b_strides, batch, splits)
    partial_args = (a, b, partials, *partial_constants)
    partial_kernel, meta = _small_partials[partial_grid](*partial_args)
    post_kernel, post_meta = _finish_splitk[post_grid](
        *post_args, num_warps=4, num_stages=1
    )
    if key is not None and not torch.cuda.is_current_stream_capturing():
        partial = bind_gemm_launch(
            _small_partials,
            partial_kernel,
            partial_grid(meta),
            partial_args,
            {},
            meta,
            3,
        )
        post = bind_gemm_launch(
            _finish_splitk, post_kernel, post_grid, post_args, {}, post_meta, 3
        )
        if partial is not None and post is not None:
            _small_launches.remember(key, (*partial, *post))
    return True
