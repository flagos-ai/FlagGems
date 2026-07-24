import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


_SAMPLED_ADDMM_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _broadcast_sparse_csr(input, shape):
    if input.shape == shape:
        return torch.sparse_csr_tensor(
            input.crow_indices().clone(),
            input.col_indices().clone(),
            input.values().clone(),
            size=input.shape,
            dtype=input.dtype,
            device=input.device,
        )
    if input.dim() != 2 or shape[-2:] != input.shape:
        raise RuntimeError(
            f"sparse_sampled_addmm: Cannot broadcast CSR tensor of shape "
            f"{tuple(input.shape)} to {tuple(shape)}"
        )
    M, N = input.shape
    batch_shape = shape[:-2]
    B = math.prod(batch_shape)
    nnz = input._nnz()

    crow = (
        input.crow_indices()
        .unsqueeze(0)
        .expand(B, M + 1)
        .reshape(batch_shape + (M + 1,))
        .contiguous()
    )
    col = (
        input.col_indices()
        .unsqueeze(0)
        .expand(B, nnz)
        .reshape(batch_shape + (nnz,))
        .contiguous()
    )
    val = (
        input.values()
        .unsqueeze(0)
        .expand(B, nnz)
        .reshape(batch_shape + (nnz,))
        .contiguous()
    )
    return torch.sparse_csr_tensor(
        crow, col, val, size=shape, dtype=input.dtype, device=input.device
    )


@libentry()
@triton.jit
def _csr_value_index_kernel(
    crow_ptr,
    col_ptr,
    idx_ptr,
    M,
    N,
    nnz_per_batch,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    b = pid // M
    r = pid % M

    crow_base = crow_ptr + b.to(tl.int64) * (M + 1)
    row_start = tl.load(crow_base + r).to(tl.int64)
    row_end = tl.load(crow_base + r + 1).to(tl.int64)

    col_base = col_ptr + b.to(tl.int64) * nnz_per_batch
    out_base = idx_ptr + b.to(tl.int64) * M * N + r.to(tl.int64) * N

    for start in range(row_start, row_end, BLOCK):
        e = start + tl.arange(0, BLOCK)
        mask = e < row_end
        c = tl.load(col_base + e, mask=mask, other=0)
        tl.store(out_base + c, e.to(tl.int32), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def sparse_sampled_addmm_kernel(
    mat1_ptr,
    mat2_ptr,
    idx_ptr,
    val_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    nnz_per_batch,
    stride_mat1_b,
    stride_mat1_m,
    stride_mat1_k,
    stride_mat2_b,
    stride_mat2_k,
    stride_mat2_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    pid = ext.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = tl.cdiv(M, BLOCK_M) * num_pid_n
    b = pid // num_tiles
    tile_id = pid % num_tiles
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    b64 = b.to(tl.int64)
    a_ptrs = (
        mat1_ptr
        + b64 * stride_mat1_b
        + offs_m[:, None] * stride_mat1_m
        + offs_k[None, :] * stride_mat1_k
    )
    b_ptrs = (
        mat2_ptr
        + b64 * stride_mat2_b
        + offs_k[:, None] * stride_mat2_k
        + offs_n[None, :] * stride_mat2_n
    )

    acc_dtype = tl.float64 if IS_FP64 else tl.float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K),
            other=0.0,
        )
        b_val = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N),
            other=0.0,
        )
        if IS_FP64:
            acc += tl.dot(a, b_val, allow_tf32=False)
        else:
            acc += tl.dot(a, b_val, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_mat1_k
        b_ptrs += BLOCK_K * stride_mat2_k

    offs_m64 = offs_m.to(tl.int64)
    offs_n64 = offs_n.to(tl.int64)
    tile_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    idx = tl.load(
        idx_ptr + b64 * M * N + offs_m64[:, None] * N + offs_n64[None, :],
        mask=tile_mask,
        other=-1,
    )
    sampled_mask = idx >= 0

    val_offs = b64 * nnz_per_batch + idx.to(tl.int64)
    old = tl.load(val_ptr + val_offs, mask=sampled_mask, other=0.0)
    new = alpha * acc + beta * old.to(acc_dtype)
    tl.store(val_ptr + val_offs, new.to(old.dtype), mask=sampled_mask)


def _dot_block(size, max_block):
    return max(16, min(triton.next_power_of_2(size), max_block))


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
            f"sparse_sampled_addmm: Expected input to be floating-point, "
            f"but got {input.dtype}"
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

    idx_map = torch.full((B, M * N), -1, dtype=torch.int32, device=input.device)

    BLOCK_M = _dot_block(M, 64)
    BLOCK_N = _dot_block(N, 64)
    BLOCK_K = _dot_block(K, 32)

    grid_fill = (B * M,)
    grid = (B * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    logger.debug(
        "GEMS SPARSE_SAMPLED_ADDMM, [shape info]: batch=%s, M=%s, N=%s, K=%s, "
        "nnz=%s, BLOCK_M=%s, BLOCK_N=%s, BLOCK_K=%s",
        B,
        M,
        N,
        K,
        nnz,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )

    with torch_device_fn.device(input.device):
        _csr_value_index_kernel[grid_fill](
            crow_2d,
            col_2d,
            idx_map,
            M,
            N,
            nnz_per_batch,
            BLOCK=256,
        )
        sparse_sampled_addmm_kernel[grid](
            mat1_f,
            mat2_f,
            idx_map,
            val_f,
            alpha,
            beta,
            M,
            N,
            K,
            nnz_per_batch,
            mat1_f.stride(0),
            mat1_f.stride(1),
            mat1_f.stride(2),
            mat2_f.stride(0),
            mat2_f.stride(1),
            mat2_f.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            IS_FP64=input.dtype == torch.float64,
            num_warps=4,
        )

    return out


def sparse_sampled_addmm(input, mat1, mat2, *, beta=1.0, alpha=1.0, out=None):
    logger.debug("GEMS SPARSE_SAMPLED_ADDMM")
    return _sparse_sampled_addmm_impl(
        input, mat1, mat2, beta=beta, alpha=alpha, out=out
    )
