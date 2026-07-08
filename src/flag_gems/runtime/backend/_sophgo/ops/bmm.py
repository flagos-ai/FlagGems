import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


def _select_bmm_tile(M: int, N: int, K: int):
    # Keep the kernel shape space intentionally small on Sophgo.
    # Start from the known-safe 32x32x32 config and only scale N for wider outputs.
    if M <= 8 and N <= 64 and K <= 256:
        return 8, 32, 32, 4
    if M <= 32 and N <= 32:
        return 32, 32, 32, 4
    if M <= 32 and N <= 64:
        return 32, 64, 32, 4
    return 32, 32, 32, 4


@libentry()
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    pid_b = tle.program_id(2)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_batch = A + pid_b * stride_ab
    b_batch = B + pid_b * stride_bb
    o_batch = O + pid_b * stride_ob

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, TILE_K)

    for k_tile in range(num_k_tiles):
        offs_k = k_tile * TILE_K + tl.arange(0, TILE_K)
        mask_k = offs_k < K

        a_ptrs = a_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_batch + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)

    o_ptrs = o_batch + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def bmm(A, B):
    logger.debug("SOPHGO BMM")

    assert A.ndim == 3 and B.ndim == 3, "bmm expects 3D tensors"
    assert A.shape[0] == B.shape[0], "batch size mismatch"
    assert A.shape[2] == B.shape[1], "inner dimension mismatch"

    batch, M, K = A.shape
    _, _, N = B.shape

    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    tile_m, tile_n, tile_k, num_warps = _select_bmm_tile(M, N, K)
    grid = (
        triton.cdiv(M, tile_m),
        triton.cdiv(N, tile_n),
        batch,
    )

    with torch_device_fn.device(A.device):
        bmm_kernel[grid](
            A,
            B,
            out,
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
            TILE_M=tile_m,
            TILE_N=tile_n,
            TILE_K=tile_k,
            num_warps=num_warps,
            num_stages=2,
        )
    return out
