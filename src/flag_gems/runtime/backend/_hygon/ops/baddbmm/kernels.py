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

"""Generic GEMM, packed GEMM and BF16 ring tuning families."""

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.libentry import LibTuner
from flag_gems.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    import triton.experimental.tle.language as tle

    from .ring import baddbmm_ring_kernel


@triton.jit
def baddbmm_kernel(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AB: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    CB: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    OB: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    STAGE_SHARED: tl.constexpr = False,
    UNROLL: tl.constexpr = 1,
    BATCH: tl.constexpr = 1,
):
    pid, batch = tl.program_id(0), tl.program_id(1)
    gm, gn = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * gn)
    size = tl.minimum(gm - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % (GROUP_M * gn) % size
    pn = pid % (GROUP_M * gn) // size
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    cols = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    ap = A + batch * AB + rows[:, None] * AM + rk[None, :] * AK
    bp = B + batch * BB + rk[:, None] * BK + cols[None, :] * BN
    if STAGE_SHARED:
        shared_b = tle.gpu.alloc(
            (BLOCK_K, BLOCK_N), B.dtype.element_ty, scope=tle.gpu.smem
        )
        local_b = tle.gpu.local_ptr(shared_b)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ki in tl.range(tl.cdiv(K, BLOCK_K), loop_unroll_factor=UNROLL):
        am = ((M % BLOCK_M == 0) | (rm < M))[:, None]
        bn = ((N % BLOCK_N == 0) | (rn < N))[None, :]
        km = (K % BLOCK_K == 0) | (rk + ki * BLOCK_K < K)
        a = tl.load(ap, am & km[None, :], 0)
        b = tl.load(bp, bn & km[:, None], 0)
        if STAGE_SHARED:
            tl.store(local_b, b)
            b = tl.load(local_b)
        acc = tl.dot(a, b, acc)
        ap += BLOCK_K * AK
        bp += BLOCK_K * BK
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    acc *= ALPHA
    if BETA != 0:
        acc += BETA * tl.load(
            Bias + batch * CB + rm[:, None] * CM + rn[None, :] * CN, mask, 0
        ).to(tl.float32)
    tl.store(Out + batch * OB + rm[:, None] * OM + rn[None, :] * ON, acc, mask)


def _prune_configs(configs, named_args, **kwargs):
    m, n, k = (named_args[name] for name in ("M", "N", "K"))
    itemsize = named_args["A"].element_size()
    result, seen = [], set()
    for config in configs:
        meta = config.kwargs
        bm, bn, bk = (meta[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        unroll = meta.get("UNROLL", 1)
        stages = config.num_stages
        if meta.get("STAGE_SHARED", False) and (not HAS_TLE or m < 256):
            continue
        if unroll >= 4 and (itemsize != 2 or k > 512):
            continue
        iters = triton.cdiv(k, bk)
        if stages > iters or unroll > iters or (iters <= 2 and unroll != 1):
            continue
        if stages > 1 and (bm + bn) * bk * itemsize * stages > 65536:
            continue
        if triton.cdiv(bm * bn, config.num_warps * 64) > 128:
            continue
        if bm > max(32, triton.next_power_of_2(m)) or bn > max(
            64, triton.next_power_of_2(n)
        ):
            continue
        signature = dict(config.all_kwargs())
        signature["GROUP_M"] = min(meta["GROUP_M"], triton.cdiv(m, bm))
        signature = tuple(sorted(signature.items()))
        if signature not in seen:
            seen.add(signature)
            result.append(config)
    return result


@triton.jit
def _packed_store(
    Bias,
    Out,
    acc,
    batch,
    rm,
    rn,
    M: tl.constexpr,
    N: tl.constexpr,
    CB: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    OB: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
):
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    acc *= ALPHA
    if BETA != 0:
        acc += BETA * tl.load(
            Bias + batch * CB + rm[:, None] * CM + rn[None, :] * CN, mask, 0
        ).to(tl.float32)
    tl.store(Out + batch * OB + rm[:, None] * OM + rn[None, :] * ON, acc, mask)


@libentry()
@triton.jit
def _pack_b(
    B,
    Packed,
    N: tl.constexpr,
    K: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    k = tl.program_id(0) * 64 + tl.arange(0, 64)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    batch = tl.program_id(2)
    mask = (k[:, None] < K) & (n[None, :] < N)
    value = tl.load(B + batch * BB + k[:, None] * BK + n[None, :] * BN, mask, 0)
    tl.store(Packed + batch * N * K + n[None, :] * K + k[:, None], value, mask)


@triton.jit
def _packed_kernel(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AB: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BB: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    CB: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    OB: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BATCH: tl.constexpr = 1,
    SPLIT_ROWS: tl.constexpr = True,
):
    pid, batch = tl.program_id(0), tl.program_id(1)
    gm, gn = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * gn)
    size = tl.minimum(gm - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % (GROUP_M * gn) % size
    pn = pid % (GROUP_M * gn) // size
    ROW_SPLIT: tl.constexpr = A.dtype.element_ty == tl.float16 or SPLIT_ROWS
    COL_SPLIT: tl.constexpr = A.dtype.element_ty == tl.float16 or not SPLIT_ROWS
    PREFETCH: tl.constexpr = A.dtype.element_ty == tl.bfloat16
    TM: tl.constexpr = BLOCK_M // 2 if ROW_SPLIT else BLOCK_M
    TN: tl.constexpr = BLOCK_N // 2 if COL_SPLIT else BLOCK_N
    rm = pm * BLOCK_M + tl.arange(0, TM)
    rn = pn * BLOCK_N + tl.arange(0, TN)
    rk = tl.arange(0, BLOCK_K)
    ap = A + batch * AB + rm[:, None] * AM + rk[None, :] * AK
    bp = B + batch * BB + rn[:, None] * BN + rk[None, :] * BK
    c00 = tl.zeros((TM, TN), tl.float32)
    if COL_SPLIT:
        c01 = tl.zeros((TM, TN), tl.float32)
    if ROW_SPLIT:
        c10 = tl.zeros((TM, TN), tl.float32)
    if ROW_SPLIT and COL_SPLIT:
        c11 = tl.zeros((TM, TN), tl.float32)
    if PREFETCH:
        km = (K % BLOCK_K == 0) | (rk + 0 * BLOCK_K < K)
        a0 = tl.load(ap, ((M % BLOCK_M == 0) | (rm[:, None] < M)) & km[None, :], 0)
        b0 = tl.trans(
            tl.load(bp, ((N % BLOCK_N == 0) | (rn[:, None] < N)) & km[None, :], 0)
        )
        if ROW_SPLIT:
            a1 = tl.load(
                ap + TM * AM,
                ((M % BLOCK_M == 0) | (rm[:, None] + TM < M)) & km[None, :],
                0,
            )
        if COL_SPLIT:
            b1 = tl.trans(
                tl.load(
                    bp + TN * BN,
                    ((N % BLOCK_N == 0) | (rn[:, None] + TN < N)) & km[None, :],
                    0,
                )
            )
        for ki in tl.range(1, tl.cdiv(K, BLOCK_K)):
            ap += BLOCK_K * AK
            bp += BLOCK_K * BK
            km = (K % BLOCK_K == 0) | (rk + ki * BLOCK_K < K)
            next_a0 = tl.load(
                ap, ((M % BLOCK_M == 0) | (rm[:, None] < M)) & km[None, :], 0
            )
            next_b0 = tl.trans(
                tl.load(bp, ((N % BLOCK_N == 0) | (rn[:, None] < N)) & km[None, :], 0)
            )
            if ROW_SPLIT:
                next_a1 = tl.load(
                    ap + TM * AM,
                    ((M % BLOCK_M == 0) | (rm[:, None] + TM < M)) & km[None, :],
                    0,
                )
            if COL_SPLIT:
                next_b1 = tl.trans(
                    tl.load(
                        bp + TN * BN,
                        ((N % BLOCK_N == 0) | (rn[:, None] + TN < N)) & km[None, :],
                        0,
                    )
                )
            c00 = tl.dot(a0, b0, c00)
            if COL_SPLIT:
                c01 = tl.dot(a0, b1, c01)
            if ROW_SPLIT:
                c10 = tl.dot(a1, b0, c10)
            if ROW_SPLIT and COL_SPLIT:
                c11 = tl.dot(a1, b1, c11)
            a0, b0 = next_a0, next_b0
            if ROW_SPLIT:
                a1 = next_a1
            if COL_SPLIT:
                b1 = next_b1
        c00 = tl.dot(a0, b0, c00)
        if COL_SPLIT:
            c01 = tl.dot(a0, b1, c01)
        if ROW_SPLIT:
            c10 = tl.dot(a1, b0, c10)
        if ROW_SPLIT and COL_SPLIT:
            c11 = tl.dot(a1, b1, c11)
    else:
        for ki in tl.range(tl.cdiv(K, BLOCK_K)):
            km = (K % BLOCK_K == 0) | (rk + ki * BLOCK_K < K)
            a0 = tl.load(ap, ((M % BLOCK_M == 0) | (rm[:, None] < M)) & km[None, :], 0)
            b0 = tl.trans(
                tl.load(bp, ((N % BLOCK_N == 0) | (rn[:, None] < N)) & km[None, :], 0)
            )
            if ROW_SPLIT:
                a1 = tl.load(
                    ap + TM * AM,
                    ((M % BLOCK_M == 0) | (rm[:, None] + TM < M)) & km[None, :],
                    0,
                )
            if COL_SPLIT:
                b1 = tl.trans(
                    tl.load(
                        bp + TN * BN,
                        ((N % BLOCK_N == 0) | (rn[:, None] + TN < N)) & km[None, :],
                        0,
                    )
                )
            c00 = tl.dot(a0, b0, c00)
            if COL_SPLIT:
                c01 = tl.dot(a0, b1, c01)
            if ROW_SPLIT:
                c10 = tl.dot(a1, b0, c10)
            if ROW_SPLIT and COL_SPLIT:
                c11 = tl.dot(a1, b1, c11)
            ap += BLOCK_K * AK
            bp += BLOCK_K * BK
    _packed_store(
        Bias, Out, c00, batch, rm, rn, M, N, CB, CM, CN, OB, OM, ON, ALPHA, BETA
    )
    if COL_SPLIT:
        _packed_store(
            Bias,
            Out,
            c01,
            batch,
            rm,
            rn + TN,
            M,
            N,
            CB,
            CM,
            CN,
            OB,
            OM,
            ON,
            ALPHA,
            BETA,
        )
    if ROW_SPLIT:
        _packed_store(
            Bias,
            Out,
            c10,
            batch,
            rm + TM,
            rn,
            M,
            N,
            CB,
            CM,
            CN,
            OB,
            OM,
            ON,
            ALPHA,
            BETA,
        )
    if ROW_SPLIT and COL_SPLIT:
        _packed_store(
            Bias,
            Out,
            c11,
            batch,
            rm + TM,
            rn + TN,
            M,
            N,
            CB,
            CM,
            CN,
            OB,
            OM,
            ON,
            ALPHA,
            BETA,
        )


def _prune_packed(configs, named_args, **kwargs):
    bf16 = named_args["A"].dtype == torch.bfloat16
    # Packed dispatch and the one-stage domain already bound K, grid size and LDS.
    return [
        config
        for config in configs
        if (bf16 or config.kwargs["SPLIT_ROWS"])
        and config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_N"]
        <= 128 * 64 * config.num_warps
    ]


@triton.jit
def _ring_kernel(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    CB: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    GROUP_M: tl.constexpr,
    BATCH: tl.constexpr = 1,
):
    baddbmm_ring_kernel(A, B, Out, Bias, CB, CM, CN, ALPHA, BETA, M, N, K, GROUP_M)


def _make_tuner(kernel, contract, prune=None, key=None):
    class _BaddbmmTuner(LibTuner.get("default")):
        def _flagtune_configs_for_mode(self, _op_name, mode):
            if mode is not runtime.TuningMode.EXPANDED:
                return self._flagtune_default_configs, self._flagtune_default_strategy
            return runtime.get_tuned_config(contract), "default"

    defaults = {
        "baddbmm": triton.Config(
            dict(
                BLOCK_M=32,
                BLOCK_N=64,
                BLOCK_K=32,
                GROUP_M=4,
                STAGE_SHARED=False,
                UNROLL=1,
            ),
            num_warps=4,
            num_stages=1,
        ),
        "baddbmm_packed": triton.Config(
            dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=4, SPLIT_ROWS=True),
            num_warps=4,
            num_stages=1,
        ),
        "baddbmm_ring": triton.Config(dict(GROUP_M=4), num_warps=8, num_stages=1),
    }
    if key is None:
        key = [
            "BATCH",
            "M",
            "N",
            "K",
            "AB",
            "AM",
            "AK",
            "BB",
            "BK",
            "BN",
            "CB",
            "CM",
            "CN",
            "OB",
            "OM",
            "ON",
            "ALPHA",
            "BETA",
        ]
    return libentry()(
        libtuner(
            configs=[defaults[contract]],
            key=key,
            strategy="default",
            prune_configs_by={"early_config_prune": prune} if prune else None,
            policy=_BaddbmmTuner,
            benchmark_mode="event",
            warmup=8,
            rep=30,
            flagtune_op_name="baddbmm",
        )(kernel)
    )


_TUNERS = {
    "matmul": _make_tuner(baddbmm_kernel, "baddbmm", _prune_configs),
    "packed": _make_tuner(_packed_kernel, "baddbmm_packed", _prune_packed),
}
if HAS_TLE:
    _TUNERS["ring"] = _make_tuner(
        _ring_kernel,
        "baddbmm_ring",
        key=["BATCH", "M", "N", "K", "CB", "CM", "CN", "ALPHA", "BETA"],
    )


@libentry()
@triton.jit
def _bias_kernel(
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    CB: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    OB: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    BETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1)
    row, col = x // N, x % N
    if BETA == 0:
        val = tl.full((BLOCK,), 0, tl.float32)
    else:
        val = (
            tl.load(Bias + batch * CB + row * CM + col * CN, x < M * N, 0).to(
                tl.float32
            )
            * BETA
        )
    tl.store(Out + batch * OB + row * OM + col * ON, val, x < M * N)
