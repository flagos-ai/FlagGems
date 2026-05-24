import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import broadcastable_to
from flag_gems.utils import triton_lang_extension as tle


ADDMM_M1_CONFIG_TABLE = (
    {"n_min": 4096, "k_min": 0, "config": (64, 8)},
    {"n_min": 2048, "k_min": 0, "config": (32, 16)},
    {"n_min": 0, "k_min": 3072, "config": (16, 16)},
    {"n_min": 0, "k_min": 0, "config": (8, 32)},
)

ADDMM_M1_TRANSPOSED_CONFIG_TABLE = (
    # Tuned on CIX P1 aarch64 (2026-03-04): BK=64 fills a full cache line.
    {"n_min": 65536, "k_min": 0, "config": (2, 64)},
    {"n_min": 2048, "k_min": 0, "config": (4, 64)},
    {"n_min": 0, "k_min": 2048, "config": (4, 64)},
    {"n_min": 0, "k_min": 0, "config": (4, 64)},
)


def _select_addmm_m1_config(N, K):
    for rule in ADDMM_M1_CONFIG_TABLE:
        if N >= rule.get("n_min", 0) and K >= rule.get("k_min", 0):
            return rule["config"]
    return 8, 32


def _select_addmm_m1_transposed_config(N, K):
    for rule in ADDMM_M1_TRANSPOSED_CONFIG_TABLE:
        if N >= rule.get("n_min", 0) and K >= rule.get("k_min", 0):
            return rule["config"]
    return 8, 32


def _is_rhs_transposed_layout(rhs):
    if rhs.ndim != 2:
        return False
    return rhs.stride(0) == 1 and rhs.stride(1) >= rhs.shape[0]


def _use_addmm_m1_transposed_fastpath_shape(N, K):
    # Avoid unstable LLVM lowering for tiny matrices on ARM cpu backend.
    return N >= 256 and K >= 256


def _use_addmm_m1_fastpath_shape(N, K):
    return N >= 256 and K >= 256


@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_m1_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_in,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_n = tle.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rk * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk < k_remaining, other=0.0)
            b = tl.load(
                b_ptrs,
                mask=(rk[:, None] < k_remaining) & (rn[None, :] < N),
                other=0.0,
            )

        a_fp = a.to(tl.float32)
        b_fp = b.to(tl.float32)
        acc += tl.sum(b_fp * a_fp[:, None], axis=0)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if beta == 0:
        out = acc * alpha
    else:
        bias_ptrs = i_ptr + rn * stride_in
        bias = tl.load(bias_ptrs, mask=rn < N, other=0.0).to(tl.float32)
        out = acc * alpha + bias * beta
    c_ptrs = c_ptr + rn * stride_cn
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=rn < N)


@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_m1_transposed_rhs_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_in,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_n = tle.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rk * stride_ak
    bt_ptrs = b_ptr + rn[:, None] * stride_bn + rk[None, :] * stride_bk
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            bt = tl.load(bt_ptrs, mask=rn[:, None] < N, other=0.0)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk < k_remaining, other=0.0)
            bt = tl.load(
                bt_ptrs,
                mask=(rn[:, None] < N) & (rk[None, :] < k_remaining),
                other=0.0,
            )

        a_fp = a.to(tl.float32)
        bt_fp = bt.to(tl.float32)
        acc += tl.sum(bt_fp * a_fp[None, :], axis=1)
        a_ptrs += BLOCK_K * stride_ak
        bt_ptrs += BLOCK_K * stride_bk

    if beta == 0:
        out = acc * alpha
    else:
        bias_ptrs = i_ptr + rn * stride_in
        bias = tl.load(bias_ptrs, mask=rn < N, other=0.0).to(tl.float32)
        out = acc * alpha + bias * beta
    c_ptrs = c_ptr + rn * stride_cn
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=rn < N)


def _launch_addmm_m1_kernel(mat1, mat2, bias, out, alpha, beta, N, K):
    block_n, block_k = _select_addmm_m1_config(N, K)
    grid = lambda META: (triton.cdiv(N, block_n),)
    addmm_m1_kernel[grid](
        mat1,
        mat2,
        bias,
        out,
        alpha,
        beta,
        N,
        K,
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        bias.stride(1),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN_K=(K % block_k == 0),
    )


def _launch_addmm_m1_transposed_rhs_kernel(mat1, mat2, bias, out, alpha, beta, N, K):
    block_n, block_k = _select_addmm_m1_transposed_config(N, K)
    grid = lambda META: (triton.cdiv(N, block_n),)
    addmm_m1_transposed_rhs_kernel[grid](
        mat1,
        mat2,
        bias,
        out,
        alpha,
        beta,
        N,
        K,
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        bias.stride(1),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN_K=(K % block_k == 0),
    )


# @libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if beta == 0:
        c = (accumulator * alpha).to(c_ptr.dtype.element_ty)
    else:
        i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
        bias = tl.load(i_ptrs, mask=c_mask, other=0.0)
        accumulator = accumulator * alpha + bias * beta
        c = accumulator.to(bias.dtype)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logging.debug("GEMS ADDMM")
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
    out_shape = (M, N)
    bias = bias.broadcast_to(out_shape)

    if M == 1 and _use_addmm_m1_fastpath_shape(N, K):
        use_fp32_m1 = (
            mat1.dtype is torch.bfloat16
            or mat2.dtype is torch.bfloat16
            or bias.dtype is torch.bfloat16
        )
        # BF16 masked_load on v8bf16 is not supported in AArch64 LLVM
        # backend (fatal "Cannot select" error in addmm_m1_kernel bias
        # tl.load). Cast all bf16 inputs to fp32 — matches the generic
        # kernel path below.
        mat1_kernel = mat1.to(torch.float32) if use_fp32_m1 else mat1
        mat2_kernel = mat2.to(torch.float32) if use_fp32_m1 else mat2
        bias_kernel = bias.to(torch.float32) if use_fp32_m1 else bias
        out_kernel = torch.empty(
            out_shape,
            device=mat1.device,
            dtype=(torch.float32 if use_fp32_m1 else mat1.dtype),
        )
        if _is_rhs_transposed_layout(mat2_kernel) and _use_addmm_m1_transposed_fastpath_shape(
            N, K
        ):
            _launch_addmm_m1_transposed_rhs_kernel(
                mat1_kernel, mat2_kernel, bias_kernel, out_kernel, alpha, beta, N, K
            )
        else:
            _launch_addmm_m1_kernel(
                mat1_kernel, mat2_kernel, bias_kernel, out_kernel, alpha, beta, N, K
            )
        return out_kernel.to(mat1.dtype) if use_fp32_m1 else out_kernel

    use_fp32_generic = (
        mat1.dtype is torch.bfloat16
        or mat2.dtype is torch.bfloat16
        or bias.dtype is torch.bfloat16
    )
    # Always cast bf16 to fp32 for the generic kernel: masked_load on bf16
    # (v8bf16) is not supported in the AArch64 LLVM backend and causes a
    # fatal "Cannot select" error.  The M=1 fastpath handles bf16 the same way.
    mat1_kernel = mat1.to(torch.float32) if use_fp32_generic else mat1
    mat2_kernel = mat2.to(torch.float32) if use_fp32_generic else mat2
    bias_kernel = bias.to(torch.float32) if use_fp32_generic else bias
    out = torch.empty(
        out_shape,
        device=mat1.device,
        dtype=(torch.float32 if use_fp32_generic else mat1.dtype),
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    addmm_kernel[grid](
        mat1_kernel,
        mat2_kernel,
        bias_kernel,
        out,
        alpha,
        beta,
        M,
        N,
        K,
        mat1_kernel.stride(0),
        mat1_kernel.stride(1),
        mat2_kernel.stride(0),
        mat2_kernel.stride(1),
        bias_kernel.stride(0),
        bias_kernel.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out.to(mat1.dtype) if use_fp32_generic else out


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logging.debug("GEMS ADDMM_OUT")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    M, K = mat1.shape
    _, N = mat2.shape

    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"

    assert broadcastable_to(bias.shape, out.shape), "Incompatible input shape"

    if mat1.stride(0) > 1 and mat1.stride(1) > 1:
        mat1 = mat1.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()
    bias = bias.broadcast_to(out.shape)

    if M == 1 and _use_addmm_m1_fastpath_shape(N, K):
        bias_kernel = bias
        use_fp32_m1 = (
            mat1.dtype is torch.bfloat16
            or mat2.dtype is torch.bfloat16
            or bias.dtype is torch.bfloat16
        )
        out_kernel = (
            torch.empty(out.shape, device=out.device, dtype=torch.float32)
            if use_fp32_m1
            else out
        )
        if _is_rhs_transposed_layout(mat2) and _use_addmm_m1_transposed_fastpath_shape(
            N, K
        ):
            _launch_addmm_m1_transposed_rhs_kernel(
                mat1, mat2, bias_kernel, out_kernel, alpha, beta, N, K
            )
        else:
            _launch_addmm_m1_kernel(
                mat1, mat2, bias_kernel, out_kernel, alpha, beta, N, K
            )
        if use_fp32_m1:
            out.copy_(out_kernel.to(out.dtype))
        return out

    use_fp32_generic = (
        mat1.dtype is torch.bfloat16
        or mat2.dtype is torch.bfloat16
        or bias.dtype is torch.bfloat16
    )
    # Always cast bf16 to fp32: see comment in addmm() above.
    mat1_kernel = mat1.to(torch.float32) if use_fp32_generic else mat1
    mat2_kernel = mat2.to(torch.float32) if use_fp32_generic else mat2
    bias_kernel = bias.to(torch.float32) if use_fp32_generic else bias
    out_kernel = (
        torch.empty(out.shape, device=out.device, dtype=torch.float32)
        if use_fp32_generic
        else out
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    addmm_kernel[grid](
        mat1_kernel,
        mat2_kernel,
        bias_kernel,
        out_kernel,
        alpha,
        beta,
        M,
        N,
        K,
        mat1_kernel.stride(0),
        mat1_kernel.stride(1),
        mat2_kernel.stride(0),
        mat2_kernel.stride(1),
        bias_kernel.stride(0),
        bias_kernel.stride(1),
        out_kernel.stride(0),
        out_kernel.stride(1),
    )
    if use_fp32_generic:
        out.copy_(out_kernel.to(out.dtype))
    return out
