import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops.sparse_sampled_addmm import _broadcast_sparse_csr
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_SAMPLED_ADDMM_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


def _get_mm_configs():
    return [
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 64}),
    ]


@libentry()
@triton.autotune(
    configs=_get_mm_configs(),
    key=["M", "N", "K"],
)
@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["TILE_K"] == 0})
@triton.jit
def _dense_bmm_kernel(
    A,
    B,
    D,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_b = ext.program_id(1)
    A += pid_b * (M * K)
    B += pid_b * (K * N)
    D += pid_b * (M * N)

    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, TILE_M)
    grid_n = tl.cdiv(N, TILE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, TILE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_n[None, :] < N), other=0.0)
        else:
            mask_k = offs_k < K - i * TILE_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

    d_ptrs = D + offs_m[:, None] * N + offs_n[None, :]
    mask_d = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(d_ptrs, acc, mask=mask_d)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _sample_rows_kernel(
    d_ptr,
    crow_ptr,
    col_ptr,
    old_ptr,
    val_ptr,
    alpha,
    beta,
    N,
    MN,
    nnz_per_batch,
    BLOCK_N: tl.constexpr,
    G: tl.constexpr,
):
    r = ext.program_id(0)
    b = ext.program_id(1)
    M = ext.num_programs(0)

    crow_base = crow_ptr + b.to(tl.int64) * (M + 1)
    row_start = tl.load(crow_base + r).to(tl.int32)
    row_end = tl.load(crow_base + r + 1).to(tl.int32)

    base = b.to(tl.int64) * nnz_per_batch
    cols = tl.arange(0, BLOCK_N)
    d_row = tl.load(
        d_ptr + b.to(tl.int64) * MN + r.to(tl.int64) * N + cols,
        mask=cols < N,
        other=0.0,
    )

    for start in range(row_start, row_end, G):
        e = start + tl.arange(0, G)
        mask = e < row_end
        c = tl.load(col_ptr + base + e, mask=mask, other=0).to(tl.int32)
        d = tl.gather(d_row, c, 0)
        old = tl.load(old_ptr + base + e, mask=mask, other=0.0)
        new = alpha * d + beta * old.to(tl.float32)
        tl.store(val_ptr + base + e, new.to(val_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _csr_row_indices_kernel(
    crow_ptr,
    row_ptr,
    M,
    nnz_per_batch,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    b = pid // M
    r = pid % M

    crow_base = crow_ptr + b.to(tl.int64) * (M + 1)
    row_start = tl.load(crow_base + r).to(tl.int32)
    row_end = tl.load(crow_base + r + 1).to(tl.int32)

    out_base = row_ptr + b.to(tl.int64) * nnz_per_batch
    for start in range(row_start, row_end, BLOCK):
        e = start + tl.arange(0, BLOCK)
        mask = e < row_end
        tl.store(out_base + e, tl.full((BLOCK,), r, dtype=tl.int32), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _sample_kernel(
    d_ptr,
    row_ptr,
    col_ptr,
    old_ptr,
    val_ptr,
    alpha,
    beta,
    N,
    MN,
    nnz_per_batch,
    BLOCK: tl.constexpr,
):
    chunk = ext.program_id(0)
    b = ext.program_id(1)

    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz_per_batch

    b64 = b.to(tl.int64)
    base = b64 * nnz_per_batch
    r = tl.load(row_ptr + base + offs, mask=mask, other=0).to(tl.int32)
    c = tl.load(col_ptr + base + offs, mask=mask, other=0).to(tl.int32)

    d = tl.load(d_ptr + b64 * MN + r * N + c, mask=mask, other=0.0)
    old = tl.load(old_ptr + base + offs, mask=mask, other=0.0)
    new = alpha * d + beta * old.to(tl.float32)
    tl.store(val_ptr + base + offs, new.to(val_ptr.dtype.element_ty), mask=mask)


def _prepare_out(input, out, out_shape, nnz_per_batch):
    if out is None:
        if input.shape == out_shape:
            out = torch.sparse_csr_tensor(
                input.crow_indices(),
                input.col_indices(),
                input.values().clone(),
                size=out_shape,
                dtype=input.dtype,
                device=input.device,
            )
            val_old = input.values()
        else:
            bcast = _broadcast_sparse_csr(input, out_shape)
            out = torch.sparse_csr_tensor(
                bcast.crow_indices(),
                bcast.col_indices(),
                bcast.values().clone(),
                size=out_shape,
                dtype=input.dtype,
                device=input.device,
            )
            val_old = bcast.values()
        return out, val_old

    if out.shape != out_shape:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected out shape {out_shape}, got {out.shape}"
        )
    if out._nnz() != nnz_per_batch:
        raise RuntimeError(
            f"sparse_sampled_addmm: Expected out nnz per batch {nnz_per_batch}, "
            f"got {out._nnz()}"
        )
    if out is input:
        return out, out.values().clone()

    if input.shape == out_shape:
        bcast = input
    else:
        bcast = _broadcast_sparse_csr(input, out_shape)
    out.crow_indices().copy_(bcast.crow_indices())
    out.col_indices().copy_(bcast.col_indices())
    out.values().copy_(bcast.values())
    val_old = bcast.values()
    return out, val_old


def _sparse_sampled_addmm_impl(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    """result = alpha * (mat1 @ mat2) * spy(input) + beta * input (CSR input)."""
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
            f"sparse_sampled_addmm: Expected input to be "
            f"float16/bfloat16/float32 on Ascend, but got {input.dtype}"
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

    out, val_old = _prepare_out(input, out, out_shape, nnz_per_batch)

    if mat1.numel() == 0 or mat2.numel() == 0 or nnz == 0 or alpha == 0.0 or K == 0:
        out.values().mul_(beta)
        return out

    if not mat1.is_contiguous():
        mat1 = mat1.contiguous()
    if not mat2.is_contiguous():
        mat2 = mat2.contiguous()
    mat1_f = mat1.reshape(B, M, K) if mat1.shape != (B, M, K) else mat1
    mat2_f = mat2.reshape(B, K, N) if mat2.shape != (B, K, N) else mat2
    val_f = out.values()
    val_old_f = val_old
    crow_2d = out.crow_indices()
    col_2d = out.col_indices()
    if not crow_2d.is_contiguous():
        crow_2d = crow_2d.contiguous()
    if not col_2d.is_contiguous():
        col_2d = col_2d.contiguous()

    logger.debug(
        "GEMS_ASCEND SPARSE_SAMPLED_ADDMM, [shape info]: batch=%s, M=%s, "
        "N=%s, K=%s, nnz=%s",
        B,
        M,
        N,
        K,
        nnz,
    )

    with torch_device_fn.device(input.device):
        D = torch.empty((B, M, N), dtype=torch.float32, device=input.device)
        grid_mm = lambda META: (
            triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
            B,
        )
        _dense_bmm_kernel[grid_mm](mat1_f, mat2_f, D, M, N, K, GROUP_M=8)

        if N <= 16384:
            _sample_rows_kernel[(M, B)](
                D,
                crow_2d,
                col_2d,
                val_old_f,
                val_f,
                alpha,
                beta,
                N,
                M * N,
                nnz_per_batch,
                BLOCK_N=triton.next_power_of_2(N),
                G=512,
            )
            return out

        row_idx = torch.empty(
            (B, nnz_per_batch), dtype=torch.int32, device=input.device
        )
        _csr_row_indices_kernel[(B * M,)](
            crow_2d,
            row_idx,
            M,
            nnz_per_batch,
            BLOCK=256,
        )
        BLOCK = 512
        _sample_kernel[(triton.cdiv(nnz_per_batch, BLOCK), B)](
            D,
            row_idx,
            col_2d,
            val_old_f,
            val_f,
            alpha,
            beta,
            N,
            M * N,
            nnz_per_batch,
            BLOCK=BLOCK,
        )

    return out


def sparse_sampled_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS_ASCEND SPARSE_SAMPLED_ADDMM")
    return _sparse_sampled_addmm_impl(
        input, mat1, mat2, beta=beta, alpha=alpha, out=out
    )
