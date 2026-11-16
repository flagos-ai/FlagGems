# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os

import torch
import triton
import triton.language as tl
from _metax.ops.addmm import (
    _MAX_SPLIT_K_PROGRAMS_PER_SM,
    _MIN_K_PER_SPLIT,
    _SPLIT_K_CANDIDATES,
    _affine_bench_post_hook,
    _affine_gemm,
    _affine_matmul,
    _affine_vector,
    _AffineTuner,
    _prune_affine_vector,
)

from flag_gems import runtime
from flag_gems.ops.mul import mul
from flag_gems.utils import get_device_properties, libentry, libtuner

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "baddbmm_metax_expand.yaml")
)
# Shared-memory requirement of the native 128-element tile.
_NATIVE128_SHARED_BYTES = 64 * 1024


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


class _BaddbmmTuner(_AffineTuner):
    """Keep pruned tiles: AABS mishandles short K and logical transpose."""

    def get_key(self, args):
        policy_key = ("confirm16x3",) if args["K"] >= 1024 else ()
        return super().get_key(args) + policy_key

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


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("baddbmm_gemm", yaml_path=EXPAND_CONFIG_FILENAME),
    key=_KEY,
    prune_configs_by={"early_config_prune": _prune_baddbmm_gemm},
    policy=_BaddbmmTuner,
    post_hook=_affine_bench_post_hook,
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
    batch = tl.program_id(1).to(tl.int64)
    _affine_gemm(
        A + batch * SAB,
        B + batch * SBB,
        C + batch * SCB,
        P + batch * SPLIT_K * M * N,
        Bias + batch * SIB,
        alpha,
        beta,
        SIM,
        SIN,
        BETA_ZERO,
        ALPHA_ONE,
        BETA_ONE,
        M,
        N,
        K,
        SAM,
        SAK,
        SBK,
        SBN,
        SCM,
        SCN,
        BM,
        BN,
        BK,
        GROUP_M,
        SPLIT_K,
        TRANSPOSE,
        STATIC_K,
        FLAT_EPILOGUE,
    )


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("baddbmm_vector", yaml_path=EXPAND_CONFIG_FILENAME),
    key=_KEY + ["TRANSPOSE"],
    prune_configs_by={"early_config_prune": _prune_affine_vector},
    policy=_BaddbmmTuner,
    post_hook=_affine_bench_post_hook,
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
    batch = tl.program_id(1).to(tl.int64)
    _affine_vector(
        A + batch * SAB,
        B + batch * SBB,
        C + batch * SCB,
        P + batch * SPLIT_K * M * N,
        Bias + batch * SIB,
        alpha,
        beta,
        SIM,
        SIN,
        BETA_ZERO,
        ALPHA_ONE,
        BETA_ONE,
        M,
        N,
        K,
        SAM,
        SAK,
        SBK,
        SBN,
        SCM,
        SCN,
        BN,
        BK,
        SPLIT_K,
        TRANSPOSE,
    )


def _baddbmm_impl(bias, a, b, beta, alpha, out=None):
    batch, m, k = a.shape
    b_batch, bk, n = b.shape
    assert batch == b_batch and k == bk, "Incompatible dimensions"
    shape = (batch, m, n)
    si = bias.broadcast_to(shape).stride()
    dtype = a.dtype
    if a.dtype != b.dtype or bias.dtype != dtype:
        raise RuntimeError("bias, batch1 and batch2 must have the same dtype")
    if out is None:
        out = torch.empty(shape, device=a.device, dtype=dtype)
    else:
        assert (
            out.shape == shape and out.dtype == dtype
        ), "Incompatible output shape or dtype for baddbmm.out"

    return _affine_matmul(
        bias,
        a,
        b,
        beta,
        alpha,
        out,
        shape,
        k,
        si,
        _baddbmm_gemm_kernel,
        _baddbmm_vector_kernel,
    )


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS_METAX BADDBMM_FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        return _baddbmm_impl(bias, A, B, beta, alpha, None)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_METAX BADDBMM_BACKWARD")
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
