import logging
import math

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

from flag_gems import runtime
from flag_gems.ops.sparse_sampled_addmm import _broadcast_sparse_csr
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


_SAMPLED_ADDMM_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


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
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def _sddmm_bmm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_wb,
    stride_wm,
    stride_wn,
    input_precision: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Batched dense GEMM tile -> workspace, following _iluvatar/ops/mm.py."""
    pid = tle.program_id(0)
    b = tle.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    b64 = b.to(tl.int64)
    A = A + b64 * stride_ab + ram[:, None] * stride_am + rk[None, :] * stride_ak
    B = B + b64 * stride_bb + rk[:, None] * stride_bk + rbn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(A)
            b_val = tl.load(B)
            acc += tl.dot(
                a, b_val, out_dtype=tl.float32, input_precision=input_precision
            )
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk
    else:
        loop_num = tl.cdiv(K, BLOCK_K) - 1
        for k in range(0, loop_num):
            a = tl.load(A)
            b_val = tl.load(B)
            acc += tl.dot(
                a, b_val, out_dtype=tl.float32, input_precision=input_precision
            )
            A += BLOCK_K * stride_ak
            B += BLOCK_K * stride_bk
        k_remaining = K - loop_num * BLOCK_K
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
        b_val = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b_val, out_dtype=tl.float32, input_precision=input_precision)

    C = C + b64 * stride_wb + rm[:, None] * stride_wm + rn[None, :] * stride_wn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc.to(C.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _gather_back_kernel(
    ws_ptr,
    crow_ptr,
    col_ptr,
    val_ptr,
    alpha,
    beta,
    M,
    N,
    nnz_per_batch,
    BLOCK: tl.constexpr,
):
    pid = tle.program_id(0)
    b = pid // M
    r = pid % M

    crow_base = crow_ptr + b.to(tl.int64) * (M + 1)
    row_start = tl.load(crow_base + r).to(tl.int64)
    row_end = tl.load(crow_base + r + 1).to(tl.int64)

    col_base = col_ptr + b.to(tl.int64) * nnz_per_batch
    ws_base = ws_ptr + b.to(tl.int64) * M * N + r.to(tl.int64) * N
    val_base = val_ptr + b.to(tl.int64) * nnz_per_batch

    for start in range(row_start, row_end, BLOCK):
        e = start + tl.arange(0, BLOCK)
        mask = e < row_end
        c = tl.load(col_base + e, mask=mask, other=0)
        w = tl.load(ws_base + c.to(tl.int64), mask=mask, other=0.0)
        old = tl.load(val_base + e, mask=mask, other=0.0)
        new = alpha * w.to(tl.float32) + beta * old.to(tl.float32)
        tl.store(val_base + e, new.to(old.dtype), mask=mask)


def _sparse_sampled_addmm_impl(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    if input.layout != torch.sparse_csr:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected input to have sparse csr layout, "
            f"but got {input.layout}"
        )
    if mat1.layout != torch.strided:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected mat1 to have strided layout, "
            f"but got {mat1.layout}"
        )
    if mat2.layout != torch.strided:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected mat2 to have strided layout, "
            f"but got {mat2.layout}"
        )
    if out is not None and out.layout != torch.sparse_csr:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected out to have sparse csr layout, "
            f"but got {out.layout}"
        )

    if input.dtype not in _SAMPLED_ADDMM_DTYPES:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected input to be one of "
            f"{_SAMPLED_ADDMM_DTYPES} on Iluvatar, but got {input.dtype}"
        )
    if input.dtype != mat1.dtype or input.dtype != mat2.dtype:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected all inputs to have the same dtype, "
            f"but got input={input.dtype}, mat1={mat1.dtype}, mat2={mat2.dtype}"
        )

    if input.dense_dim() != 0:
        raise RuntimeError("sparse_sampled_addmm: Expected non-hybrid input tensor")
    if out is not None and out.dense_dim() != 0:
        raise RuntimeError("sparse_sampled_addmm: Expected non-hybrid out tensor")

    if mat1.dim() < 2 or mat2.dim() < 2:
        raise RuntimeError(
            "sparse_sampled_addmm: Expected mat1 and mat2 to be at least 2-D matrices"
        )

    batch_dims = mat1.shape[:-2]
    M, K = mat1.shape[-2:]
    N = mat2.shape[-1]

    if mat2.shape[:-2] != batch_dims:
        raise RuntimeError(
            "sparse_sampled_addmm: Expected mat1 and mat2 to have the same batch size"
        )
    if input.dim() > 2 and input.shape[:-2] != batch_dims:
        raise RuntimeError(
            "sparse_sampled_addmm: Expected input and mat1 to have the same batch size"
        )
    if input.shape[-2] != M or input.shape[-1] != N:
        raise RuntimeError(
            "sparse_sampled_addmm: input.shape[-2:] must match (M, N) of mat1 @ mat2"
        )
    if mat2.shape[-2] != K:
        raise RuntimeError(
            "sparse_sampled_addmm: mat1 and mat2 shapes cannot be multiplied"
        )

    out_shape = batch_dims + (M, N)
    B = math.prod(batch_dims) if batch_dims else 1

    nnz_per_batch = input._nnz()
    nnz = nnz_per_batch * B

    if out is None:
        out = _broadcast_sparse_csr(input, out_shape)
    else:
        if out.shape != out_shape:
            raise RuntimeError(
                f"sparse_sampled_addmm: Expected out shape {out_shape}, got {out.shape}"
            )
        if out._nnz() != nnz_per_batch:
            raise RuntimeError(
                f"sparse_sampled_addmm: Expected out nnz per batch {nnz_per_batch}, "
                f"got {out._nnz()}"
            )
        if out is not input:
            out.copy_(_broadcast_sparse_csr(input, out_shape))

    if mat1.numel() == 0 or mat2.numel() == 0 or nnz == 0 or alpha == 0.0 or K == 0:
        out.values().mul_(beta)
        return out

    mat1_f = mat1.contiguous().reshape(B, M, K)
    mat2_f = mat2.contiguous().reshape(B, K, N)
    val_f = out.values().reshape(B * nnz_per_batch)
    crow_2d = out.crow_indices().reshape(B, M + 1).contiguous()
    col_2d = out.col_indices().reshape(B, nnz_per_batch).contiguous()

    ws = torch.empty((B, M, N), dtype=torch.float32, device=input.device)

    input_precision = "ieee" if input.dtype == torch.float32 else None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        B,
    )

    with torch_device_fn.device(input.device):
        _sddmm_bmm_kernel[grid](
            mat1_f,
            mat2_f,
            ws,
            M,
            N,
            K,
            mat1_f.stride(0),
            mat1_f.stride(1),
            mat1_f.stride(2),
            mat2_f.stride(0),
            mat2_f.stride(1),
            mat2_f.stride(2),
            ws.stride(0),
            ws.stride(1),
            ws.stride(2),
            input_precision=input_precision,
        )
        _gather_back_kernel[(B * M,)](
            ws,
            crow_2d,
            col_2d,
            val_f,
            alpha,
            beta,
            M,
            N,
            nnz_per_batch,
            BLOCK=256,
            num_warps=4,
        )

    return out


def sparse_sampled_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0):
    logger.debug("GEMS SPARSE_SAMPLED_ADDMM")
    return _sparse_sampled_addmm_impl(input, mat1, mat2, beta=beta, alpha=alpha)


def sparse_sampled_addmm_out(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS SPARSE_SAMPLED_ADDMM_OUT")
    if out is None:
        raise TypeError("sparse_sampled_addmm(): out must be provided for out variant")
    return _sparse_sampled_addmm_impl(
        input, mat1, mat2, beta=beta, alpha=alpha, out=out
    )
