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

"""Hygon fused baddbmm kernels with shape- and dtype-based dispatch.

Large half-precision products pack B into a K-contiguous layout. FlagTune
selects accumulator decomposition, tiling, pipelining and loop unrolling.
Small outputs use GEMV or cooperative split-K. All temporary tensors belong
to the invocation; tuning and compilation caches are managed by FlagGems.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.libentry import LibTuner
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)

if HAS_TLE:
    import triton.experimental.tle.language as tle


@triton.jit
def _product(
    B,
    ap,
    bp,
    am,
    bn,
    rk,
    K: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGE_SHARED: tl.constexpr,
    UNROLL: tl.constexpr,
    ITERS: tl.constexpr,
    EVEN_K: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if STAGE_SHARED:
        # The B-only variant is the only explicit staging mode retained. The
        # earlier A/B/both matrix multiplied LDS traffic and did not show a
        # stable benefit on gfx936; B-only has a measured win for the large-M
        # model cases and remains a normal FlagTune candidate.
        sb = tle.gpu.alloc((BLOCK_K, BN), B.dtype.element_ty, scope=tle.gpu.smem)
        lb = tle.gpu.local_ptr(sb)
    for k in tl.range(ITERS, loop_unroll_factor=UNROLL, num_stages=LOOP_STAGES):
        a = tl.load(
            ap,
            am[:, None] & (EVEN_K | (rk[None, :] + k * BLOCK_K < K)),
            0,
        )
        b = tl.load(
            bp,
            bn[None, :] & (EVEN_K | (rk[:, None] + k * BLOCK_K < K)),
            0,
        )
        if STAGE_SHARED:
            tl.store(lb, b)
            b = tl.load(lb)
        # Leave dot lowering to the Hygon Triton backend. The official lean
        # baddbmm path uses the same default and avoids forcing a schedule or
        # precision lowering that may block hardware-specific optimization.
        acc = tl.dot(a, b, acc)
        ap += BLOCK_K * AK
        bp += BLOCK_K * BK
    return acc


@libentry()
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
    STAGE_SHARED: tl.constexpr,
    UNROLL: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    PART_STRIDE: tl.constexpr = 0,
    Counter=None,
    FinalOut=None,
    COOPERATIVE: tl.constexpr = False,
    FOB: tl.constexpr = 0,
    FOM: tl.constexpr = 0,
    FON: tl.constexpr = 0,
    BATCH: tl.constexpr = 1,
    LOOP_STAGES: tl.constexpr = 1,
    WRAP_M: tl.constexpr = False,
):
    pid, batch = tl.program_id(0), tl.program_id(1)
    gm, gn = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * gn)
    size = tl.minimum(gm - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % (GROUP_M * gn) % size
    pn = pid % (GROUP_M * gn) // size
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    # Contiguous tiles are common in the model benchmark.  These are the
    # same address-contiguity facts used by the Hygon mm kernel; they let the
    # backend form vectorized loads without changing the masked tail path.
    if M % BLOCK_M == 0:
        rm_load = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    elif WRAP_M:
        # Padded rows read valid A rows; only valid output rows are stored.
        # Do not assert alignment on a modulo with a ragged M.
        rm_load = rm % M
    else:
        rm_load = rm
    if N % BLOCK_N == 0:
        rn_load = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rn_load = rn
    iters: tl.constexpr = triton.cdiv(K, BLOCK_K * SPLIT_K)
    rk = tl.arange(0, BLOCK_K) + tl.program_id(2) * iters * BLOCK_K
    ap = A + batch * AB + rm_load[:, None] * AM + rk[None, :] * AK
    bp = B + batch * BB + rk[:, None] * BK + rn_load[None, :] * BN
    acc = _product(
        B,
        ap,
        bp,
        WRAP_M | (M % BLOCK_M == 0) | (rm < M),
        (N % BLOCK_N == 0) | (rn < N),
        rk,
        K,
        AK,
        BK,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        STAGE_SHARED,
        UNROLL,
        iters,
        K % (BLOCK_K * SPLIT_K) == 0,
        LOOP_STAGES,
    )
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if COOPERATIVE and SPLIT_K > 1:
        base = Out + batch * OB + rm[:, None] * OM + rn[None, :] * ON
        tl.store(base + tl.program_id(2) * PART_STRIDE, acc, mask)
        # Publish every partial tile before issuing one CTA ticket. There is
        # one counter per output tile; the narrow-output split-K path does not
        # need a fixed cache-line-sized slot for every counter.
        tl.debug_barrier()
        counter_tiles = tl.cdiv(M, 16) * tl.cdiv(N, 16)
        ticket = tl.atomic_add(Counter + batch * counter_tiles + pid, 1, sem="acq_rel")
        if ticket == SPLIT_K - 1:
            total = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
            for part in tl.static_range(SPLIT_K):
                total += tl.load(base + part * PART_STRIDE, mask, 0)
            total *= ALPHA
            if BETA != 0:
                total += (
                    tl.load(
                        Bias + batch * CB + rm[:, None] * CM + rn[None, :] * CN, mask, 0
                    ).to(tl.float32)
                    * BETA
                )
            tl.store(
                FinalOut + batch * FOB + rm[:, None] * FOM + rn[None, :] * FON,
                total,
                mask,
            )
            # No other CTA accesses this counter again in this invocation.
            # The next invocation on this workspace is ordered on the stream.
            tl.store(Counter + batch * counter_tiles + pid, 0)
    else:
        acc = acc * ALPHA
        if BETA != 0:
            bias = tl.load(
                Bias + batch * CB + rm[:, None] * CM + rn[None, :] * CN, mask, 0
            ).to(tl.float32)
            acc = acc + bias * BETA
        if COOPERATIVE:
            # SPLIT_K=1 is a candidate in the narrow-output search too.
            dst = FinalOut + batch * FOB + rm[:, None] * FOM + rn[None, :] * FON
        else:
            dst = Out + batch * OB + rm[:, None] * OM + rn[None, :] * ON
        tl.store(dst, acc, mask)


def _prune_configs(configs, named_args, **kwargs):
    """Remove configurations with no plausible win for this concrete shape.

    Narrow-output searches include SPLIT_K=1, allowing the fused ordinary
    epilogue to compete with cooperative reduction on the same real shape.
    """
    m, n, k = named_args["M"], named_args["N"], named_args["K"]
    a = named_args["A"]
    batch = int(a.shape[0])
    cu_count = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Avoid retaining a high-pressure tile solely because every faster-looking
    # candidate was filtered.  The fallback is only used for unusual shapes
    # (for example, a direct short-K split-K correctness launch).
    fallback = None
    max_bm = max(16, triton.next_power_of_2(m))
    max_bn = max(16, triton.next_power_of_2(n))
    ACCUMULATOR_LIMIT = 128
    itemsize = named_args["A"].element_size()
    result, seen = [], set()
    for config in configs:
        meta = config.kwargs
        stage = meta.get("STAGE_SHARED", 0)
        if stage and not HAS_TLE:
            continue
        bm, bn, bk = (meta[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        # Wider reduction tiles and deep unrolling trade operand registers for
        # fewer short-reduction loop instructions. Keep this search family in
        # half precision, where operand storage is compact enough.
        if (bk >= 256 or meta.get("UNROLL", 1) >= 4 or config.num_warps == 2) and (
            itemsize != 2 or k > 512
        ):
            continue
        split = meta.get("SPLIT_K", 1)
        iters = triton.cdiv(k, bk * split)
        loop_stages = meta.get("LOOP_STAGES", 1)
        # Keep loop and launch stage counts consistent.
        if "LOOP_STAGES" in meta and config.num_stages != loop_stages:
            continue
        if loop_stages > iters:
            continue
        # Conservative LDS bound for compiler-managed pipelining.
        if loop_stages > 1 and (bm + bn) * bk * itemsize * loop_stages > 65536:
            continue
        if stage and loop_stages != 1:
            continue
        if meta.get("WRAP_M", False) and m % bm == 0:
            continue

        # This conservative gfx936 estimate filters obvious VGPR pressure
        # before compilation; it is intentionally looser than a hard limit.
        acc_per_lane = triton.cdiv(bm * bn, config.num_warps * 64)
        if acc_per_lane > ACCUMULATOR_LIMIT:
            continue

        # B-only LDS staging helps ordinary GEMM only at large M. The one
        # narrow-output candidate is Split-K with a two-way, 64-wide K
        # partition, which limits staging and reduction overhead.
        if stage and split == 1 and m < 256:
            continue
        if stage and split > 1 and (split != 2 or bk != 64):
            continue
        if (
            stage
            and split > 1
            and (
                bm != 64
                or bn != 128
                or meta["GROUP_M"] != 1
                or config.num_warps != 4
                or meta.get("sched_latency") != "none"
            )
        ):
            continue
        if stage and bn * bk * itemsize > 65536:
            continue

        # Keep every split partition non-empty, including the fallback used by
        # unusual direct correctness launches.
        if split > 1 and (split - 1) * iters * bk >= k:
            continue

        # Record a structurally valid candidate before performance heuristics
        # below. It guarantees that an unusual direct launch cannot produce an
        # empty tuner set.
        if fallback is None:
            fallback = config

        # A tile larger than the problem mostly computes masked lanes. Keep a
        # 16-wide minimum so tiny legal dimensions still have a candidate.
        if bm > max_bm or bn > max_bn:
            continue

        # Keep split-K near the occupancy boundary; SPLIT_K=1 remains
        # available so the tuner can measure the reduction tradeoff.
        if split > 1:
            output_tiles = batch * triton.cdiv(m, bm) * triton.cdiv(n, bn)
            if output_tiles >= 2 * cu_count:
                continue
            # Include split=2 even for severely underfilled launches: fully
            # covering a wave is not worth an arbitrarily expensive reduction.
            if split > 2 and output_tiles * (split // 2) >= 2 * cu_count:
                continue
            # A split CTA should perform enough dot iterations to amortize its
            # partial output and cooperative reduction protocol.
            if iters < 4:
                continue

        # Unrolling a loop shorter than the requested factor has no benefit;
        # for one or two iterations it only adds scheduling overhead.
        unroll = meta.get("UNROLL", 1)
        if unroll > iters or (iters <= 2 and unroll != 1):
            continue
        signature = dict(config.all_kwargs())
        # GROUP_M >= the number of M tiles gives the same tile ordering.
        signature["GROUP_M"] = min(meta["GROUP_M"], triton.cdiv(m, bm))
        signature = tuple(sorted(signature.items()))
        if signature not in seen:
            seen.add(signature)
            result.append(config)
    if not result:
        if fallback is None:
            raise RuntimeError("No feasible Hygon baddbmm configuration")
        result.append(fallback)
    return result


def _default_config(contract):
    # Default execution must not expand the performance search. FlagTune loads
    # the full backend domain separately when the user enables tuning.
    if contract == "baddbmm_gemv":
        return triton.Config(dict(BLOCK_N=32, BLOCK_K=64), num_warps=4, num_stages=1)
    if contract in ("baddbmm_packed", "baddbmm_bf16_packed"):
        bf16 = contract == "baddbmm_bf16_packed"
        return triton.Config(
            dict(
                BLOCK_M=128,
                BLOCK_N=256 if bf16 else 128,
                BLOCK_K=64 if bf16 else 32,
                GROUP_M=4,
                QUADRANT=1 if bf16 else False,
                UNROLL=1,
                LOOP_STAGES=1,
                PREFETCH=False,
            ),
            num_warps=4,
            num_stages=1,
        )
    meta = dict(
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_M=4,
        STAGE_SHARED=0,
        UNROLL=1,
        LOOP_STAGES=1,
        WRAP_M=False,
        sched_latency="none",
        matrix_instr_nonkdim=0,
    )
    if contract == "baddbmm_rect":
        meta["TRANSPOSE_NM"] = False
    else:
        meta["SPLIT_K"] = 1
    return triton.Config(meta, num_warps=4, num_stages=1)


_TUNING_KEY = [
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


def _make_tuner(kernel, contract, prune):
    key = _TUNING_KEY + (["FOB", "FOM", "FON"] if contract == "baddbmm_splitk" else [])

    class _BaddbmmTuner(LibTuner.get("default")):
        def _flagtune_configs_for_mode(self, _op_name, mode):
            if mode is not runtime.TuningMode.EXPANDED:
                return self._flagtune_default_configs, self._flagtune_default_strategy
            return runtime.get_tuned_config(contract), "default"

    return libentry()(
        libtuner(
            configs=[_default_config(contract)],
            key=key,
            strategy="default",
            prune_configs_by={"early_config_prune": prune},
            policy=_BaddbmmTuner,
            benchmark_mode="event",
            # Hygon event timings have visible sub-0.1 ms jitter. Match the
            # acceptance benchmark's longer measurement window so FlagTune
            # does not select a transiently fast candidate.
            warmup=8,
            rep=30,
            flagtune_op_name="baddbmm",
        )(kernel)
    )


# Separate algorithm tuners specialize the shared GEMM body at compile time.
# The cooperative variant owns SPLIT_K and its partial-reduction dataflow.
_autotuned_matmul = _make_tuner(baddbmm_kernel.fn, "baddbmm", _prune_configs)
_autotuned_splitk = _make_tuner(baddbmm_kernel.fn, "baddbmm_splitk", _prune_configs)


@libentry()
@triton.jit
def _gemv_kernel(
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BATCH: tl.constexpr = 1,
):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row, batch = tl.program_id(1) % M, tl.program_id(1) // M
    rk = tl.arange(0, BLOCK_K)
    # Keep the multiply in a 2-D tile so the backend can vectorize the
    # K-reduction; on gfx936 this maps better than materializing a reduction
    # after every K tile for the long-K GEMV cases.
    acc = tl.full((BLOCK_K, BLOCK_N), 0, tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start * BLOCK_K + rk
        a = tl.load(A + batch * AB + row * AM + k * AK, k < K, 0).to(tl.float32)
        b = tl.load(
            B + batch * BB + k[:, None] * BK + rn[None, :] * BN,
            (k[:, None] < K) & (rn[None, :] < N),
            0,
        )
        acc += a[:, None] * b.to(tl.float32)
    val = tl.sum(acc, 0) * ALPHA
    if BETA != 0:
        val += (
            tl.load(Bias + batch * CB + row * CM + rn * CN, rn < N, 0).to(tl.float32)
            * BETA
        )
    tl.store(Out + batch * OB + row * OM + rn * ON, val, rn < N)


def _prune_gemv_configs(configs, named_args, **kwargs):
    # The accumulator is FP32 even for half inputs.
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_K"] * config.kwargs["BLOCK_N"]
        <= config.num_warps * 64 * 256
    ]


_autotuned_gemv = _make_tuner(_gemv_kernel.fn, "baddbmm_gemv", _prune_gemv_configs)


# Rectangular kernels may transpose tile axes to improve operand reuse.
@triton.jit
def _rect_kernel(
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
    STAGE_SHARED: tl.constexpr,
    UNROLL: tl.constexpr,
    BATCH: tl.constexpr = 1,
    LOOP_STAGES: tl.constexpr = 1,
    WRAP_M: tl.constexpr = False,
    TRANSPOSE_NM: tl.constexpr = False,
):
    # Reorient the dot and output strides without allocating transposed tensors.
    Rows: tl.constexpr = N if TRANSPOSE_NM else M
    Cols: tl.constexpr = M if TRANSPOSE_NM else N
    LeftBatch: tl.constexpr = BB if TRANSPOSE_NM else AB
    LeftRow: tl.constexpr = BN if TRANSPOSE_NM else AM
    LeftK: tl.constexpr = BK if TRANSPOSE_NM else AK
    RightBatch: tl.constexpr = AB if TRANSPOSE_NM else BB
    RightK: tl.constexpr = AK if TRANSPOSE_NM else BK
    RightCol: tl.constexpr = AM if TRANSPOSE_NM else BN
    BiasRow: tl.constexpr = CN if TRANSPOSE_NM else CM
    BiasCol: tl.constexpr = CM if TRANSPOSE_NM else CN
    OutRow: tl.constexpr = ON if TRANSPOSE_NM else OM
    OutCol: tl.constexpr = OM if TRANSPOSE_NM else ON
    if TRANSPOSE_NM:
        Left, Right = B, A
    else:
        Left, Right = A, B
    pid, batch = tl.program_id(0), tl.program_id(1)
    gm, gn = tl.cdiv(Rows, BLOCK_M), tl.cdiv(Cols, BLOCK_N)
    group = pid // (GROUP_M * gn)
    size = tl.minimum(gm - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % (GROUP_M * gn) % size
    pn = pid % (GROUP_M * gn) // size
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    # Contiguous tiles are common in the model benchmark.  These are the
    # same address-contiguity facts used by the Hygon mm kernel; they let the
    # backend form vectorized loads without changing the masked tail path.
    if Rows % BLOCK_M == 0:
        rm_load = tl.max_contiguous(tl.multiple_of(rm % Rows, BLOCK_M), BLOCK_M)
    elif WRAP_M:
        # Padded rows read valid Left rows; only valid output rows are stored.
        # Do not assert alignment on a modulo with a ragged Rows.
        rm_load = rm % Rows
    else:
        rm_load = rm
    if Cols % BLOCK_N == 0:
        rn_load = tl.max_contiguous(tl.multiple_of(rn % Cols, BLOCK_N), BLOCK_N)
    else:
        rn_load = rn
    iters: tl.constexpr = triton.cdiv(K, BLOCK_K)
    rk = tl.arange(0, BLOCK_K)
    ap = Left + batch * LeftBatch + rm_load[:, None] * LeftRow + rk[None, :] * LeftK
    bp = Right + batch * RightBatch + rk[:, None] * RightK + rn_load[None, :] * RightCol
    acc = _product(
        Right,
        ap,
        bp,
        WRAP_M | (Rows % BLOCK_M == 0) | (rm < Rows),
        (Cols % BLOCK_N == 0) | (rn < Cols),
        rk,
        K,
        LeftK,
        RightK,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        STAGE_SHARED,
        UNROLL,
        iters,
        K % BLOCK_K == 0,
        LOOP_STAGES,
    )
    mask = (rm[:, None] < Rows) & (rn[None, :] < Cols)
    acc *= ALPHA
    if BETA != 0:
        bias = tl.load(
            Bias + batch * CB + rm[:, None] * BiasRow + rn[None, :] * BiasCol, mask, 0
        ).to(tl.float32)
        acc += bias * BETA
    tl.store(Out + batch * OB + rm[:, None] * OutRow + rn[None, :] * OutCol, acc, mask)


def _prune_rect_configs(configs, named_args, **kwargs):
    result = []
    for transpose in (False, True):
        group = [c for c in configs if c.kwargs["TRANSPOSE_NM"] == transpose]
        facts = dict(named_args)
        if transpose:
            facts["M"], facts["N"] = facts["N"], facts["M"]
        result.extend(_prune_configs(group, facts, **kwargs))
    return result


_autotuned_rect = _make_tuner(_rect_kernel, "baddbmm_rect", _prune_rect_configs)


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
    UNROLL: tl.constexpr,
    BATCH: tl.constexpr = 1,
    LOOP_STAGES: tl.constexpr = 1,
    QUADRANT: tl.constexpr = False,
    PREFETCH: tl.constexpr = False,
):
    pid, batch = tl.program_id(0), tl.program_id(1)
    gm, gn = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * gn)
    size = tl.minimum(gm - group * GROUP_M, GROUP_M)
    pm = group * GROUP_M + pid % (GROUP_M * gn) % size
    pn = pid % (GROUP_M * gn) // size
    ROW_SPLIT: tl.constexpr = QUADRANT == 1 or QUADRANT == 2
    COL_SPLIT: tl.constexpr = QUADRANT == 1 or QUADRANT == 3
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
        for ki in tl.range(
            1, tl.cdiv(K, BLOCK_K), num_stages=LOOP_STAGES, loop_unroll_factor=UNROLL
        ):
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
        for ki in tl.range(
            tl.cdiv(K, BLOCK_K), num_stages=LOOP_STAGES, loop_unroll_factor=UNROLL
        ):
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


def _prune_bf16_packed(configs, named_args, **kwargs):
    m, n, k = (named_args[name] for name in ("M", "N", "K"))
    batch = named_args["A"].shape[0]
    result, seen = [], set()
    for config in configs:
        meta = config.kwargs
        bm, bn, bk = (meta[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        stages = meta["LOOP_STAGES"]
        if stages != config.num_stages or stages > triton.cdiv(k, bk):
            continue
        if bm * bn > 128 * 64 * config.num_warps:
            continue
        # Stage one can reuse compiler-managed LDS operand buffers. The
        # compiler reports actual allocation/resource failure after pruning.
        live = (max(bm, bn) if stages == 1 else (bm + bn) * stages) * bk * 2
        if live > 65536:
            continue
        if m * n >= 1048576 and batch * triton.cdiv(m, bm) * triton.cdiv(n, bn) < 40:
            continue
        if meta["UNROLL"] > triton.cdiv(k, bk):
            continue
        signature = tuple(sorted(config.all_kwargs().items()))
        if signature not in seen:
            seen.add(signature)
            result.append(config)
    return result or configs[:1]


_autotuned_bf16_packed = _make_tuner(
    _packed_kernel, "baddbmm_bf16_packed", _prune_bf16_packed
)


def _prune_packed(configs, named_args, **kwargs):
    # Independent dtype keys are provided by LibTuner's benchmark protocol.
    result = []
    for config in configs:
        bm, bn, bk = (config.kwargs[x] for x in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        stages = config.kwargs["LOOP_STAGES"]
        if stages != config.num_stages:
            continue
        if (bm + bn) * bk * 2 * stages > 65536:
            continue
        if bm * bn // (64 * config.num_warps) > 128:
            continue
        result.append(config)
    return result


_autotuned_packed = _make_tuner(_packed_kernel, "baddbmm_packed", _prune_packed)


def _algorithm(batch, m, n, k, device=None, dtype=None):
    # Amortize packing across many row tiles for half-precision inputs.
    if (
        dtype in (torch.float16, torch.bfloat16)
        and batch == 1
        and m >= 4096
        and n >= 1024
        and k >= 1024
    ):
        return "bf16_packed" if dtype == torch.bfloat16 else "packed"

    # Long-K, tall rectangular GEMMs with an N tail use a separate search.
    if batch == 1 and m >= 4096 and m >= 2 * n and k >= 2048 and n % 128:
        return "rect"
    if m == 1 or m * n <= 16:
        return "gemv"
    # Near one wave, compare ordinary GEMM and split-K in the same tuner.
    # Count the actual tiled grid, including M padding and batch separately.
    cu_count = torch.cuda.get_device_properties(device).multi_processor_count
    output_tiles = batch * triton.cdiv(m, 64) * triton.cdiv(n, 128)
    if (
        1 < m <= 128
        and k >= 2048
        and output_tiles < 2 * cu_count
        and batch * m * n <= 1048576
    ):
        return "splitk"
    return "matmul"


_TUNERS = {
    "bf16_packed": _autotuned_bf16_packed,
    "packed": _autotuned_packed,
    "matmul": _autotuned_matmul,
    "rect": _autotuned_rect,
    "splitk": _autotuned_splitk,
    "gemv": _autotuned_gemv,
}


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


def _validate(bias, a, b):
    if (
        a.ndim != 3
        or b.ndim != 3
        or a.shape[0] != b.shape[0]
        or a.shape[2] != b.shape[1]
    ):
        raise RuntimeError("baddbmm expects compatible batches of matrices")
    if a.dtype != b.dtype or a.dtype != bias.dtype:
        raise RuntimeError("baddbmm expects all inputs to have the same dtype")
    if a.device != b.device or a.device != bias.device:
        raise RuntimeError("baddbmm expects all inputs on the same device")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "Hygon baddbmm supports float16, bfloat16 and float32"
        )
    shape = (a.shape[0], a.shape[1], b.shape[2])
    if bias.ndim > 3:
        raise RuntimeError("baddbmm bias is not broadcastable to the output")
    sizes = (1,) * (3 - bias.ndim) + tuple(bias.shape)
    strides = (0,) * (3 - bias.ndim) + bias.stride()
    if any(size != 1 and size != target for size, target in zip(sizes, shape)):
        raise RuntimeError("baddbmm bias is not broadcastable to the output")
    return tuple(0 if size == 1 else stride for size, stride in zip(sizes, strides))


def _split_workspace(device, split_k, batch, m, n):
    # Each invocation owns fresh counters; the last CTA restores them to 0.
    # The smallest supported tile is 16x16, which bounds the number of output
    # tile counters even though FlagTune may choose larger tiles.
    return (
        torch.empty((split_k, batch, m, n), device=device, dtype=torch.float32),
        torch.zeros(
            (batch * triton.cdiv(m, 16) * triton.cdiv(n, 16),),
            device=device,
            dtype=torch.int32,
        ),
    )


def _launch(bias, a, b, beta, alpha, out, bias_strides=None):
    if bias_strides is None:
        bias_strides = _validate(bias, a, b)
    batch, m, k = a.shape
    n = b.shape[2]
    if not batch or not m or not n:
        return out
    with torch_device_fn.device(a.device):
        if k == 0 or alpha == 0:
            _bias_kernel[(triton.cdiv(m * n, 256), batch)](
                bias, out, m, n, *bias_strides, *out.stride(), beta, 256
            )
            return out
        algorithm = _algorithm(batch, m, n, k, a.device, a.dtype)
        if algorithm == "gemv":
            _autotuned_gemv[lambda meta: (triton.cdiv(n, meta["BLOCK_N"]), batch * m)](
                a,
                b,
                bias,
                out,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *bias_strides,
                *out.stride(),
                alpha,
                beta,
                BATCH=batch,
            )
            return out

        if algorithm in ("packed", "bf16_packed"):
            packed = torch.empty((batch, n, k), device=b.device, dtype=b.dtype)
            _pack_b[(triton.cdiv(k, 64), triton.cdiv(n, 64), batch)](
                b, packed, n, k, *b.stride(), num_warps=4
            )
            b = packed.transpose(1, 2)

        cooperative = algorithm == "splitk"
        counters = None
        partial = out
        if cooperative:
            # Allocate enough fresh storage for every supported split count.
            partial, counters = _split_workspace(a.device, 32, batch, m, n)
        grid = lambda meta: (
            (
                triton.cdiv(n, meta["BLOCK_M"]) * triton.cdiv(m, meta["BLOCK_N"])
                if meta.get("TRANSPOSE_NM", False)
                else triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"])
            ),
            batch,
            meta.get("SPLIT_K", 1),
        )
        launch_args = (
            a,
            b,
            bias,
            partial,
            m,
            n,
            k,
            *a.stride(),
            *b.stride(),
            *bias_strides,
            *partial.stride()[-3:],
            alpha,
            beta,
        )
        launch_kwargs = {"BATCH": batch}
        if algorithm not in ("packed", "bf16_packed", "rect"):
            launch_kwargs.update(
                PART_STRIDE=batch * m * n if cooperative else 0,
                Counter=counters,
                FinalOut=out,
                COOPERATIVE=cooperative,
                FOB=out.stride(0),
                FOM=out.stride(1),
                FON=out.stride(2),
            )
        _TUNERS[algorithm][grid](*launch_args, **launch_kwargs)
    return out


class _Baddbmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, a, b, beta, alpha):
        ctx.save_for_backward(a, b)
        ctx.bias_shape, ctx.alpha, ctx.beta = bias.shape, alpha, beta
        bias_strides = _validate(bias, a, b)
        out = torch.empty(
            (a.shape[0], a.shape[1], b.shape[2]), device=a.device, dtype=a.dtype
        )
        return _launch(bias, a, b, beta, alpha, out, bias_strides=bias_strides)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        from flag_gems.ops.copy import copy_
        from flag_gems.ops.mul import mul
        from flag_gems.ops.sum_to_size import sum_to_size

        def matmul_grad(left, right):
            dtype = left.dtype
            if dtype == torch.float16:
                left_fp32 = torch.empty_like(left, dtype=torch.float32)
                right_fp32 = torch.empty_like(right, dtype=torch.float32)
                copy_(left_fp32, left)
                copy_(right_fp32, right)
                left, right = left_fp32, right_fp32
            # Reuse this operator's strided Triton product with a fused scale.
            # beta=0 keeps the uninitialized scalar bias unread.
            bias = torch.empty((), device=left.device, dtype=left.dtype)
            result = torch.empty(
                (left.shape[0], left.shape[1], right.shape[2]),
                device=left.device,
                dtype=left.dtype,
            )
            _launch(bias, left, right, 0.0, ctx.alpha, result)
            if dtype == torch.float16:
                output = torch.empty_like(result, dtype=dtype)
                return copy_(output, result)
            return result

        dbias = (
            sum_to_size(mul(grad, ctx.beta), ctx.bias_shape)
            if ctx.needs_input_grad[0]
            else None
        )
        da = matmul_grad(grad, b.transpose(1, 2)) if ctx.needs_input_grad[1] else None
        db = matmul_grad(a.transpose(1, 2), grad) if ctx.needs_input_grad[2] else None
        return dbias, da, db, None, None


def _call(bias, a, b, beta, alpha):
    if torch.is_grad_enabled() and any(x.requires_grad for x in (bias, a, b)):
        return _Baddbmm.apply(bias, a, b, beta, alpha)
    bias_strides = _validate(bias, a, b)
    output_shape = (a.shape[0], a.shape[1], b.shape[2])
    if bias.ndim == 3 and tuple(bias.shape) == output_shape and bias.is_contiguous():
        out = torch.empty_like(bias)
    else:
        out = torch.empty(output_shape, dtype=a.dtype, device=a.device)
    return _launch(bias, a, b, beta, alpha, out, bias_strides=bias_strides)


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    logger.debug("GEMS BADDBMM")
    return _call(bias, A, B, beta, alpha)


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    logger.debug("GEMS BADDBMM_OUT")
    _validate(bias, A, B)
    if torch.is_grad_enabled() and any(x.requires_grad for x in (bias, A, B, out)):
        raise RuntimeError("baddbmm.out does not support automatic differentiation")
    if (
        out.shape != (A.shape[0], A.shape[1], B.shape[2])
        or out.dtype != A.dtype
        or out.device != A.device
    ):
        raise RuntimeError(
            "baddbmm.out expects the output shape, dtype and device to match"
        )
    if any(torch._C._overlaps(out, x) for x in (bias, A, B)):
        result = _call(bias, A, B, beta, alpha)
        from flag_gems.ops.copy import copy_

        copy_(out, result)
        return out
    return _launch(bias, A, B, beta, alpha, out)
