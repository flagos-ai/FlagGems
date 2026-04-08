import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm"),
    key=["M", "N", "K"],
    strategy=["log", "log", "log"],
)
# @triton.heuristics(runtime.get_heuristic_config("bmm"))
@triton.jit
def bmm_kernel(
    A_in,
    B_in,
    O_in,
    Batch,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_GRID_DIM: tl.constexpr,
    num_warps: tl.constexpr,
):
    pid_bmn = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    grid_num = tl.cdiv(MAX_GRID_DIM, num_warps)
    for pid in tl.range(pid_bmn, Batch * grid_m * grid_n, grid_num):
        pid_b = pid // (grid_m * grid_n)
        pid_m = (pid % (grid_m * grid_n)) // grid_n
        pid_n = pid % grid_n

        A = A_in + pid_b * M * K
        B = B_in + pid_b * K * N
        O_out = O_in + pid_b * M * N

        L_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        R_block_ptr = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=O_out,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(
                L_block_ptr,
                boundary_check=(
                    0,
                    1,
                ),
                padding_option="zero",
            )
            b = tl.load(
                R_block_ptr,
                boundary_check=(
                    0,
                    1,
                ),
                padding_option="zero",
            )
            acc += tl.dot(a, b, out_dtype=tl.float32)
            L_block_ptr = tl.advance(L_block_ptr, (0, BLOCK_K))
            R_block_ptr = tl.advance(R_block_ptr, (BLOCK_K, 0))
        c = acc.to(O_out.dtype.element_ty)
        tl.store(
            O_block_ptr,
            c,
            boundary_check=(
                0,
                1,
            ),
        )


def bmm(A, B):
    logger.debug("GEMS BMM")
    Batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((Batch, M, N), dtype=A.dtype, device=A.device)

    MAX_GRID_DIM = 24
    grid = lambda META: (
        min(
            triton.cdiv(MAX_GRID_DIM, META["num_warps"]),
            Batch * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        ),
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid](
            A,
            B,
            out,
            Batch,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            MAX_GRID_DIM=MAX_GRID_DIM,
        )
    return out


def bmm_out(A, B, out):
    logger.debug("GEMS BMM_OUT")
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    Batch, M, K = A.shape
    _, _, N = B.shape

    A = A.contiguous()
    B = B.contiguous()
    MAX_GRID_DIM = 24
    grid = lambda META: (
        min(
            triton.cdiv(MAX_GRID_DIM, META["num_warps"]),
            Batch * triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        ),
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid](
            A,
            B,
            out,
            Batch,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            MAX_GRID_DIM=MAX_GRID_DIM,
        )
    return out
