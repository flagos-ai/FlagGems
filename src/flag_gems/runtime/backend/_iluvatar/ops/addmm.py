import logging
import math

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

from .mm import _to_tl_type, get_higher_dtype

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 15,
    },
    warmup=5,
    rep=10,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        "UPGRADE": lambda args: math.ceil(
            (args["M"] * args["N"]) / (args["BLOCK_M"] * args["BLOCK_N"])
        ).bit_length()
        > 31,
        "UPGRADE_A_OFFS": lambda args: math.ceil(args["M"] * args["K"]).bit_length()
        > 31,
        "UPGRADE_B_OFFS": lambda args: math.ceil(args["K"] * args["N"]).bit_length()
        > 31,
        "UPGRADE_C_OFFS": lambda args: math.ceil(args["M"] * args["N"]).bit_length()
        > 31,
    }
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    A,
    B,
    bias_ptr,
    C,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    acc_dtype: tl.constexpr,
    input_precision: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    UPGRADE_C_OFFS: tl.constexpr,
):
    if UPGRADE:
        pid = tle.program_id(0)
        pid_z = tle.program_id(1)
    else:
        pid = tl.program_id(0)
        pid_z = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # index ranges for A, B
    if UPGRADE_A_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        ram = (tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)).to(tl.int64)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if UPGRADE_B_OFFS:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        rbn = (tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)).to(tl.int64)
    else:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk
    else:
        loop_num = tl.cdiv(K, BLOCK_K * SPLIT_K) - 1
        for k in range(0, loop_num):
            a = tl.load(A)
            b = tl.load(B)
            if AB_DTYPE is not None:
                a = a.to(AB_DTYPE)
                b = b.to(AB_DTYPE)
            if fp8_fast_accum:
                acc = tl.dot(
                    a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
                )
            else:
                acc += tl.dot(
                    a, b, out_dtype=acc_dtype, input_precision=input_precision
                )
            A += BLOCK_K * SPLIT_K * stride_ak
            B += BLOCK_K * SPLIT_K * stride_bk

        _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
        k_remaining = K - loop_num * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if fp8_fast_accum:
            acc = tl.dot(
                a, b, acc, out_dtype=acc_dtype, input_precision=input_precision
            )
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)

    # rematerialize rm, rn for C and bias
    if UPGRADE_C_OFFS:
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn).to(tl.int64)
        bias_ptr = bias_ptr + (rm[:, None] * stride_im + rn[None, :] * stride_in).to(
            tl.int64
        )
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        bias_ptr = bias_ptr + (rm[:, None] * stride_im + rn[None, :] * stride_in)

    mask = (rm < M)[:, None] & (rn < N)[None, :]
    bias = tl.load(bias_ptr, mask=mask, other=0.0)

    acc = acc * alpha + tl.where(pid_z == 0, bias.to(tl.float32) * beta, 0.0)
    c = acc.to(C.dtype.element_ty)
    if SPLIT_K == 1:
        tl.store(C, c, mask=mask)
    else:
        tl.atomic_add(C, c, mask=mask)


def _launch_addmm(mat1, mat2, bias, out, M, N, K, alpha, beta):
    """Launch Triton addmm kernel; out must be pre-allocated."""
    ab_dtype = get_higher_dtype(mat1.dtype, mat2.dtype)
    acc_dtype_tl = tl.float32
    ab_dtype_tl = _to_tl_type(ab_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )

    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
            acc_dtype=acc_dtype_tl,
            input_precision=None,
            fp8_fast_accum=True,
            GROUP_M=8,
            AB_DTYPE=ab_dtype_tl,
        )
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("ILUVATAR GEMS ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()

    out_dtype = get_higher_dtype(mat1.dtype, mat2.dtype)
    out = torch.empty((M, N), device=mat1.device, dtype=out_dtype)
    bias = bias.broadcast_to(out.shape).contiguous()

    return _launch_addmm(mat1, mat2, bias, out, M, N, K, alpha, beta)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("ILUVATAR GEMS ADDMM_OUT")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape
    if out is None:
        out = torch.empty(
            (M, N), device=mat1.device, dtype=get_higher_dtype(mat1.dtype, mat2.dtype)
        )
    else:
        assert out.shape == (M, N), "Incompatible output shape"

    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()
    bias = bias.broadcast_to(out.shape).contiguous()

    return _launch_addmm(mat1, mat2, bias, out, M, N, K, alpha, beta)
