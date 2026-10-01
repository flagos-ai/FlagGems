# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mul import mul
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import get_device_properties, libentry, libtuner
from flag_gems.utils.libentry import LibTuner

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "baddbmm_metax_expand.yaml")
)
# Shared-memory requirement of the native 128-element tile.
_NATIVE128_SHARED_BYTES = 64 * 1024

# Bound split-K partition size, workspace, and concurrent programs.
_SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16, 32)
_MIN_K_PER_SPLIT = 256
_MAX_SPLIT_K_WORKSPACE_BYTES = 32 * 1024**2
_MAX_SPLIT_K_PROGRAMS_PER_SM = 4
_MAX_VECTOR_SPLIT_K_PROGRAMS_PER_SM = 8


# BATCH, MAX_SPLIT and *_ALIGNMENT are also host tuning/cache inputs.
# BATCH sizes the benchmark finish grid even though kernel programs obtain
# their batch index from program_id. Keep these fields in both signatures.
_KEY = [
    "BATCH",
    "SAB",
    "SBB",
    "SCB",
    "SIB",
    "M",
    "N",
    "K",
    "SAM",
    "SAK",
    "SBK",
    "SBN",
    "SCM",
    "SCN",
    "SIM",
    "SIN",
    "BETA_ZERO",
    "ALPHA_ONE",
    "BETA_ONE",
    "MAX_SPLIT",
    "A_ALIGNMENT",
    "B_ALIGNMENT",
    "C_ALIGNMENT",
    "BIAS_ALIGNMENT",
]


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_finish_kernel(
    P,
    C,
    Bias,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    SCB: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    SIB: tl.constexpr,
    SIM: tl.constexpr,
    SIN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA_ONE: tl.constexpr,
    ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m, n = i // N, i % N
    acc = tl.zeros((BLOCK,), tl.float32)
    if not ZERO:
        for split in tl.static_range(SPLIT_K):
            acc += tl.load(P + (batch * SPLIT_K + split) * M * N + i, i < M * N, 0)
        if not ALPHA_ONE:
            acc *= alpha
    if not BETA_ZERO:
        bias = tl.load(Bias + batch * SIB + m * SIM + n * SIN, i < M * N, 0)
        bias = bias.to(tl.float32)
        acc += bias if BETA_ONE else beta * bias
    tl.store(C + batch * SCB + m * SCM + n * SCN, acc, i < M * N)


def _baddbmm_bench_post_hook(args, exception):
    # Include the selected split-K reduction in candidate timing.
    if exception is None and args["SPLIT_K"] > 1:
        _baddbmm_finish_kernel[
            (triton.cdiv(args["M"] * args["N"], 256), args["BATCH"])
        ](
            args["P"],
            args["C"],
            args["Bias"],
            args["alpha"],
            args["beta"],
            args["M"],
            args["N"],
            args["SCB"],
            args["SCM"],
            args["SCN"],
            args["SIB"],
            args["SIM"],
            args["SIN"],
            args["SPLIT_K"],
            args["BETA_ZERO"],
            args["ALPHA_ONE"],
            args["BETA_ONE"],
            False,
            BLOCK=256,
            num_warps=4,
        )


class _BaddbmmTuner(LibTuner.get("default")):
    """Keep pruned tiles: AABS mishandles short K and logical transpose."""

    def get_key(self, args):
        policy_key = ("confirm16x3",) if args["K"] >= 1024 else ()
        return super().get_key(args) + policy_key

    def _bench(self, *args, config, **meta):
        options = {**meta, **config.all_kwargs()}
        values = {**dict(zip(self.arg_names, args)), **options}

        def launch():
            self.fn.run(*args, **options)
            self.post_hook(values, exception=None)

        try:
            return self.do_bench(launch, quantiles=(0.5, 0.2, 0.8))
        except triton.runtime.errors.OutOfResources:
            return [float("inf")] * 3

    def policy(self, bench_fn, configs, args, kwargs):
        best, timings = super().policy(bench_fn, configs, args, kwargs)
        values = {**dict(zip(self.arg_names, args)), **kwargs}
        if values["K"] < 1024:
            return best, timings
        finalists = [
            config
            for config in sorted(timings, key=timings.get)[:16]
            if timings[config][0] < float("inf")
        ]
        if len(finalists) < 2:
            return best, timings
        # Recheck long reductions in alternating full-call rounds.
        # Coarse benchmark data stays reusable; get_key versions the selection.
        samples = {config: [] for config in finalists}
        with self.use_benchmark_protocol("replay", warmup=0, rep=100):
            for iteration in range(3):
                order = finalists if iteration % 2 == 0 else reversed(finalists)
                for config in order:
                    value = self._bench(*args, config=config, **kwargs)
                    if isinstance(value, (int, float)):
                        value = [value] * 3
                    samples[config].append(value)
        for config, values in samples.items():
            timings[config] = [sorted(v[i] for v in values)[1] for i in range(3)]
        return min(finalists, key=timings.get), timings


def _prune_baddbmm_gemm(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = args["M"], args["N"], args["K"]
    properties = get_device_properties(args["A"].device.index)
    sm = properties.multi_processor_count
    shared_bytes = properties.shared_memory_per_block
    result = []
    for config in configs:
        meta = config.kwargs
        bm, bn, bk = meta["BM"], meta["BN"], meta["BK"]
        native_tile = (
            bm == bn == bk == 128
            and config.num_warps == 4
            and config.num_stages == 4
            and meta["pipeline"] == "cpasync"
            and not meta["scenario"]
            and not meta["TRANSPOSE"]
            and not meta["STATIC_K"]
        )
        dense_output = args["SCM"] == n and args["SCN"] == 1
        direct = (
            native_tile
            and args["A"].dtype in (torch.float16, torch.bfloat16)
            and args["C"].dtype == args["A"].dtype
            and m >= 1024
            and n >= 128
            and k >= 512
            and args["SAM"] == k
            and args["SAK"] == 1
            and dense_output
            and all(args[name].data_ptr() % 16 == 0 for name in ("A", "B", "C"))
            and k % 8 == n % 8 == 0
            and args["BATCH"] * triton.cdiv(m, bm) * triton.cdiv(n, bn) >= sm // 2
        )
        direct_nn = direct and args["SBK"] == n and args["SBN"] == 1
        direct_nt = direct and args["SBK"] == 1 and args["SBN"] == k
        # Flatten before bias arithmetic so the NT half store does not
        # require the faulty MMA-layout bias/store conversion.
        if meta["FLAT_EPILOGUE"] and not direct_nt:
            continue
        mt, nt = (n, m) if meta["TRANSPOSE"] else (m, n)
        if bm > max(32, triton.next_power_of_2(mt)) or bn > max(
            32, triton.next_power_of_2(nt)
        ):
            continue
        if min(m, n) >= 1024 and bm * bn < 4096:
            continue
        shared = (
            _NATIVE128_SHARED_BYTES
            if direct_nn or (direct_nt and meta["FLAT_EPILOGUE"])
            else (bm + bn)
            * bk
            * args["A"].element_size()
            * (1 if config.num_stages == 1 else 2)
        )
        if shared > shared_bytes:
            continue
        if (
            bk > max(32, triton.next_power_of_2(k))
            or meta["scenario"] == "reduceSmemUsage"
        ):
            continue
        if meta["scenario"] == "unprefetch" and (mt % bm or nt % bn or k % bk):
            continue
        if meta["STATIC_K"] and (k > 128 or min(m, n) < 64):
            continue
        if bm >= 256 and bn >= 256 and args["SBK"] == 1:
            continue
        if meta["TRANSPOSE"] and not (args["SBN"] == 1 or args["SAM"] == 1):
            continue
        # Roll changes only the installed compiler's loop-unroll policy.
        if meta["scenario"] == "roll" and not (64 <= min(m, n) <= 256 and k >= 1024):
            continue
        tiles = args["BATCH"] * triton.cdiv(mt, bm) * triton.cdiv(nt, bn)
        for split in _SPLIT_K_CANDIDATES:
            if split > args["MAX_SPLIT"]:
                break
            if (meta["FLAT_EPILOGUE"] or direct_nn) and split != 1:
                continue
            if split > 1 and (
                meta["STATIC_K"]
                or k < split * _MIN_K_PER_SPLIT
                or tiles * split > _MAX_SPLIT_K_PROGRAMS_PER_SM * sm
            ):
                continue
            if (
                args["A"].dtype == torch.float32
                and split > 1
                and k % (bk * split)
                and meta["pipeline"].startswith("cpasync")
            ):
                continue
            candidate = copy.deepcopy(config)
            candidate.kwargs["SPLIT_K"] = split
            result.append(candidate)
    return result


def _prune_baddbmm_vector(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    n = args["M"] if args["TRANSPOSE"] else args["N"]
    sk = args["SAK"] if args["TRANSPOSE"] else args["SBK"]
    sn = args["SAM"] if args["TRANSPOSE"] else args["SBN"]
    along_k = sk == 1 or sk <= sn or n < 32
    rows = args["N"] if args["TRANSPOSE"] else args["M"]
    sm = get_device_properties(args["A"].device.index).multi_processor_count
    result = []
    for config in configs:
        if config.kwargs["BN"] > max(1, triton.next_power_of_2(n)):
            continue
        if not (
            (along_k and config.kwargs["BN"] <= 8)
            or (not along_k and config.kwargs["BN"] >= 32)
        ):
            continue
        tiles = args["BATCH"] * rows * triton.cdiv(n, config.kwargs["BN"])
        for split in _SPLIT_K_CANDIDATES:
            if split > args["MAX_SPLIT"]:
                break
            # Include the extra SIMT wave needed by rounded-up K partitions.
            if split > 1 and (
                args["K"] < split * _MIN_K_PER_SPLIT
                or tiles * split > _MAX_VECTOR_SPLIT_K_PROGRAMS_PER_SM * sm
            ):
                continue
            candidate = copy.deepcopy(config)
            candidate.kwargs["SPLIT_K"] = split
            result.append(candidate)
    return result


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("baddbmm_gemm", yaml_path=EXPAND_CONFIG_FILENAME),
    key=_KEY,
    prune_configs_by={"early_config_prune": _prune_baddbmm_gemm},
    policy=_BaddbmmTuner,
    post_hook=_baddbmm_bench_post_hook,
    flagtune_op_name="baddbmm",
    flagtune_expand_op_name="baddbmm_gemm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=30,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_gemm_kernel(
    A,
    B,
    C,
    P,
    Bias,
    alpha,
    beta,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SBB: tl.constexpr,
    SCB: tl.constexpr,
    SIB: tl.constexpr,
    SIM: tl.constexpr,
    SIN: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA_ONE: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    A_ALIGNMENT: tl.constexpr,
    B_ALIGNMENT: tl.constexpr,
    C_ALIGNMENT: tl.constexpr,
    BIAS_ALIGNMENT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr = 1,
    SPLIT_K: tl.constexpr = 1,
    TRANSPOSE: tl.constexpr = False,
    STATIC_K: tl.constexpr = False,
    FLAT_EPILOGUE: tl.constexpr = False,
):
    """One output tile per CTA, optionally with a deterministic K partition."""
    if SPLIT_K > 1:
        C = P
        SCM, SCN = N, 1
        SCB = SPLIT_K * M * N
    batch = tl.program_id(1).to(tl.int64)
    A += batch * SAB
    B += batch * SBB
    C += batch * SCB
    Bias += batch * SIB
    if TRANSPOSE:
        # Compute C^T = B^T A^T. Swapping the dot operands changes which
        # operand feeds which MMA port without materializing a transpose.
        A, B = B, A
        M, N = N, M
        SAM, SAK, SBK, SBN = SBN, SBK, SAK, SAM
        SCM, SCN = SCN, SCM
        SIM, SIN = SIN, SIM
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    split = tl.program_id(0) // (nm * nn) % SPLIT_K
    pid = tl.program_id(0) % (nm * nn)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    group_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    pm = first_m + local % group_m
    pn = local // group_m
    mi = pm * BM + tl.arange(0, BM)
    ni = pn * BN + tl.arange(0, BN)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    ki = tl.arange(0, BK) + split * iterations * BK
    # Tell the vectorizer each axis is a dense power-of-two tile. The values
    # do not change; masked tails still compare against M/N/K below.
    mi = tl.max_contiguous(tl.multiple_of(mi, BM), BM)
    ni = tl.max_contiguous(tl.multiple_of(ni, BN), BN)
    ki = tl.max_contiguous(tl.multiple_of(ki, BK), BK)
    ap = A + mi[:, None].to(tl.int64) * SAM + ki[None, :].to(tl.int64) * SAK
    bp = B + ki[:, None].to(tl.int64) * SBK + ni[None, :].to(tl.int64) * SBN
    acc = tl.zeros((BM, BN), tl.float32)
    if STATIC_K:
        # Short reductions cannot amortize a pipelined loop's prologue and
        # epilogue. Keep addresses in int64 here too, including sliced views.
        for k in tl.static_range((K + BK * SPLIT_K - 1) // (BK * SPLIT_K)):
            if K % (BK * SPLIT_K) == 0:
                if M % BM == 0:
                    ak = tl.load(ap)
                else:
                    ak = tl.load(ap, mi[:, None] < M, other=0)
                if N % BN == 0:
                    bk = tl.load(bp)
                else:
                    bk = tl.load(bp, ni[None, :] < N, other=0)
            else:
                ak = tl.load(
                    ap, (mi[:, None] < M) & (ki[None, :] + k * BK < K), other=0
                )
                bk = tl.load(
                    bp, (ki[:, None] + k * BK < K) & (ni[None, :] < N), other=0
                )
            acc = tl.dot(ak, bk, acc, out_dtype=tl.float32, allow_tf32=False)
            ap += BK * SAK
            bp += BK * SBK
    else:
        for k in range(iterations):
            if K % (BK * SPLIT_K) == 0:
                if M % BM == 0:
                    ak = tl.load(ap)
                else:
                    ak = tl.load(ap, mi[:, None] < M, other=0)
                if N % BN == 0:
                    bk = tl.load(bp)
                else:
                    bk = tl.load(bp, ni[None, :] < N, other=0)
            else:
                ak = tl.load(
                    ap, (mi[:, None] < M) & (ki[None, :] + k * BK < K), other=0
                )
                bk = tl.load(
                    bp, (ki[:, None] + k * BK < K) & (ni[None, :] < N), other=0
                )
            acc = tl.dot(ak, bk, acc, out_dtype=tl.float32, allow_tf32=False)
            ap += BK * SAK
            bp += BK * SBK
    if FLAT_EPILOGUE:
        # Materialize the accumulator's output order before the bias arithmetic.
        i = tl.arange(0, BM * BN)
        rows = (pm * BM + i // BN).to(tl.int64)
        cols = (pn * BN + i % BN).to(tl.int64)
        value = tl.reshape(acc, (BM * BN,))
        mask = (rows < M) & (cols < N)
        if SPLIT_K == 1:
            if not ALPHA_ONE:
                value *= alpha
            if not BETA_ZERO:
                bias = tl.load(Bias + rows * SIM + cols * SIN, mask, 0).to(tl.float32)
                value += bias if BETA_ONE else beta * bias
        tl.store(C + split.to(tl.int64) * M * N + rows * SCM + cols * SCN, value, mask)
    else:
        cp = (
            C
            + split.to(tl.int64) * M * N
            + mi[:, None].to(tl.int64) * SCM
            + ni[None, :].to(tl.int64) * SCN
        )
        if SPLIT_K == 1:
            if not ALPHA_ONE:
                acc *= alpha
            if not BETA_ZERO:
                if SIM == 0 and SIN == 0:
                    bias = tl.full((BM, BN), tl.load(Bias).to(tl.float32), tl.float32)
                elif SIM == 0:
                    bias = tl.broadcast_to(
                        tl.load(
                            Bias + ni[None, :].to(tl.int64) * SIN, ni[None, :] < N, 0
                        ).to(tl.float32),
                        (BM, BN),
                    )
                elif SIN == 0:
                    bias = tl.broadcast_to(
                        tl.load(
                            Bias + mi[:, None].to(tl.int64) * SIM, mi[:, None] < M, 0
                        ).to(tl.float32),
                        (BM, BN),
                    )
                else:
                    bias = tl.load(
                        Bias
                        + mi[:, None].to(tl.int64) * SIM
                        + ni[None, :].to(tl.int64) * SIN,
                        (mi[:, None] < M) & (ni[None, :] < N),
                        other=0,
                    ).to(tl.float32)
                acc += bias if BETA_ONE else beta * bias
        tl.store(cp, acc, (mi[:, None] < M) & (ni[None, :] < N))


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("baddbmm_vector", yaml_path=EXPAND_CONFIG_FILENAME),
    key=_KEY + ["TRANSPOSE"],
    prune_configs_by={"early_config_prune": _prune_baddbmm_vector},
    policy=_BaddbmmTuner,
    post_hook=_baddbmm_bench_post_hook,
    flagtune_op_name="baddbmm",
    flagtune_expand_op_name="baddbmm_vector",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    rep=30,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_vector_kernel(
    A,
    B,
    C,
    P,
    Bias,
    alpha,
    beta,
    BATCH: tl.constexpr,
    SAB: tl.constexpr,
    SBB: tl.constexpr,
    SCB: tl.constexpr,
    SIB: tl.constexpr,
    SIM: tl.constexpr,
    SIN: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    ALPHA_ONE: tl.constexpr,
    BETA_ONE: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SAM: tl.constexpr,
    SAK: tl.constexpr,
    SBK: tl.constexpr,
    SBN: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
    A_ALIGNMENT: tl.constexpr,
    B_ALIGNMENT: tl.constexpr,
    C_ALIGNMENT: tl.constexpr,
    BIAS_ALIGNMENT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    if SPLIT_K > 1:
        C = P
        SCM, SCN = N, 1
        SCB = SPLIT_K * M * N
    batch = tl.program_id(1).to(tl.int64)
    A += batch * SAB
    B += batch * SBB
    C += batch * SCB
    Bias += batch * SIB
    if TRANSPOSE:
        A, B = B, A
        M, N = N, M
        SAM, SAK, SBK, SBN = SBN, SBK, SAK, SAM
        SCM, SCN = SCN, SCM
        SIM, SIN = SIN, SIM
    nn = tl.cdiv(N, BN)
    pid = tl.program_id(0)
    split = pid // (M * nn)
    m = (pid // nn % M).to(tl.int64)
    n = (pid % nn * BN + tl.arange(0, BN)).to(tl.int64)
    iterations = tl.cdiv(K, BK * SPLIT_K)
    k = tl.arange(0, BK).to(tl.int64) + split * iterations * BK
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(iterations):
        ks = k + start * BK
        a = tl.load(A + m * SAM + ks * SAK, ks < K, 0).to(tl.float32)
        b = tl.load(
            B + n[:, None] * SBN + ks[None, :] * SBK,
            (n[:, None] < N) & (ks[None, :] < K),
            0,
        ).to(tl.float32)
        acc = tl.fma(a[None, :], b, acc)
    result = tl.sum(acc, 1)
    if SPLIT_K == 1:
        if not ALPHA_ONE:
            result *= alpha
        if not BETA_ZERO:
            bias = tl.load(Bias + m * SIM + n * SIN, n < N, 0).to(tl.float32)
            result += bias if BETA_ONE else bias * beta
    tl.store(C + split.to(tl.int64) * M * N + m * SCM + n * SCN, result, n < N)


def _baddbmm_impl(bias, a, b, beta, alpha, out=None):
    dtype = a.dtype
    batch, m, n = a.shape[0], a.shape[-2], b.shape[-1]
    shape = (batch, m, n)
    si = bias.broadcast_to(shape).stride()
    if out is None:
        out = torch.empty(shape, device=a.device, dtype=dtype)

    k = a.shape[-1]
    if not batch or not m or not n:
        return out
    with torch_device_fn.device(a.device):
        if not k or alpha == 0:
            _baddbmm_finish_kernel[(triton.cdiv(m * n, 256), batch)](
                out,
                out,
                bias,
                alpha,
                beta,
                m,
                n,
                *out.stride(),
                *si,
                1,
                beta == 0,
                alpha == 1,
                beta == 1,
                True,
                BLOCK=256,
                num_warps=4,
            )
            return out

        small_strided = (
            max(m, n) <= 32
            and k <= 256
            and (1 not in a.stride()[-2:] or 1 not in b.stride()[-2:])
        )
        transpose = not small_strided and (n == 1 or (n <= 8 and m > n))
        rows, cols = (n, m) if transpose else (m, n)
        vector = small_strided or rows == 1 or (rows <= 8 and cols <= 32)
        # Split-K partials are FP32 even when the final output is half precision.
        bytes_per_split = batch * m * n * torch.float32.itemsize
        split_limit = min(
            _SPLIT_K_CANDIDATES[-1],
            max(1, k // _MIN_K_PER_SPLIT),
            max(1, _MAX_SPLIT_K_WORKSPACE_BYTES // bytes_per_split),
        )
        max_split = (
            1
            if small_strided
            else max(split for split in _SPLIT_K_CANDIDATES if split <= split_limit)
        )
        partial = torch.empty(
            (max_split * batch * m * n if max_split > 1 else 0,),
            dtype=torch.float32,
            device=a.device,
        )
        args = (
            a,
            b,
            out,
            partial,
            bias,
            alpha,
            beta,
            batch,
            a.stride(0),
            b.stride(0),
            out.stride(0),
            *si,
            beta == 0,
            alpha == 1,
            beta == 1,
            max_split,
            m,
            n,
            k,
            *a.stride()[-2:],
            *b.stride()[-2:],
            *out.stride()[-2:],
            a.data_ptr() % 16,
            b.data_ptr() % 16,
            out.data_ptr() % 16,
            bias.data_ptr() % 16,
        )
        if vector:
            grid = lambda meta: (
                rows * triton.cdiv(cols, meta["BN"]) * meta["SPLIT_K"],
                batch,
            )
            _, meta = _baddbmm_vector_kernel[grid](*args, TRANSPOSE=transpose)
        else:

            def grid(meta):
                mt, nt = (n, m) if meta["TRANSPOSE"] else (m, n)
                return (
                    triton.cdiv(mt, meta["BM"])
                    * triton.cdiv(nt, meta["BN"])
                    * meta["SPLIT_K"],
                    batch,
                )

            _, meta = _baddbmm_gemm_kernel[grid](*args)
        if meta["SPLIT_K"] > 1:
            # All FP32 partials are ready before the reduction and affine epilogue.
            _baddbmm_finish_kernel[(triton.cdiv(m * n, 256), batch)](
                partial,
                out,
                bias,
                alpha,
                beta,
                m,
                n,
                *out.stride(),
                *si,
                meta["SPLIT_K"],
                beta == 0,
                alpha == 1,
                beta == 1,
                False,
                BLOCK=256,
                num_warps=4,
            )
    return out


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS METAX BADDBMM FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        return _baddbmm_impl(bias, A, B, beta, alpha, None)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS METAX BADDBMM BACKWARD")
        A, B, bias = ctx.saved_tensors

        grad_A = None
        grad_B = None
        grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_bias = compute_bias_grad(grad_output, ctx.beta, bias)
        if ctx.needs_input_grad[1]:
            grad_A = compute_A_grad(grad_output, B, ctx.alpha)
        if ctx.needs_input_grad[2]:
            grad_B = compute_B_grad(A, grad_output, ctx.alpha)

        return grad_bias, grad_A, grad_B, None, None


def compute_bias_grad(d_output, beta, bias):
    grad_bias = mul(d_output, beta)
    if grad_bias.shape != bias.shape:
        # Sum over broadcasted dimensions
        while grad_bias.dim() > bias.dim():
            grad_bias = grad_bias.sum(dim=0)
        for i in range(bias.dim()):
            if bias.shape[i] == 1 and grad_bias.shape[i] > 1:
                grad_bias = grad_bias.sum(dim=i, keepdim=True)
    return grad_bias.view(bias.shape)


def _bmm_no_tf32(lhs, rhs):
    # The metax Triton bmm kernel is inaccurate for large K and cannot
    # compile when the contraction dim is < 16, so fall back to torch.bmm.
    # TF32 must be disabled to match the fp32 precision expected by the tests
    # (the Triton kernel uses allow_tf32=False).
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.bmm(lhs, rhs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def compute_A_grad(d_output, B, alpha):
    B_T = B.transpose(1, 2)
    if B.dtype == torch.float16:
        mul1 = _bmm_no_tf32(d_output.to(torch.float32), B_T.to(torch.float32))
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = _bmm_no_tf32(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        mul2 = _bmm_no_tf32(A_T.to(torch.float32), d_output.to(torch.float32))
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = _bmm_no_tf32(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    return _baddbmm_impl(bias, A, B, beta, alpha, out)


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return BaddbmmFunction.apply(bias, A, B, beta, alpha)
