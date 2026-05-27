import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt

from flag_gems import runtime
from flag_gems.fused import outer  # noqa: E402
from flag_gems.ops import mul  # noqa: E402
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm_spacemit"),
    key=["M", "N", "K"],
)
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
    stride_cb,
    stride_cm,
    stride_cn,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    TILE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    SUB_BLK_K: tl.constexpr,
):
    pidx = tl.program_id(0)
    pidy = tl.program_id(1)
    pid_b = tl.program_id(2)

    pid_m = pidx
    pid_n = pidy

    block_m = pid_m * TILE_M
    block_n = pid_n * TILE_N

    offset_a = pid_b * stride_ab
    offset_b = pid_b * stride_bb
    offset_o = pid_b * stride_cb

    a_ptr = tl.make_block_ptr(
        A + offset_a,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(block_m, 0),
        block_shape=(TILE_M, TILE_K),
        order=(1, 0),
    )

    b_ptr = tl.make_block_ptr(
        B + offset_b,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, block_n),
        block_shape=(TILE_K, TILE_N),
        order=(1, 0),
    )

    o_ptr = tl.make_block_ptr(
        O + offset_o,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(block_m, block_n),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    if EVEN_K:
        a_descriptor_load = smt.descriptor_load(a_ptr, (0, 0))
        a = smt.view(a_descriptor_load, (0, 0), (TILE_M, TILE_K), (MICRO_M, MICRO_K))
        b_descriptor_load = smt.descriptor_load(b_ptr, (0, 0))
        b = smt.view(b_descriptor_load, (0, 0), (TILE_K, TILE_N), (MICRO_K, MICRO_N))
        acc = smt.dot(a, b)
    else:
        acc = tl.zeros((TILE_M, TILE_N), dtype=A.type.element_ty)
        acc = smt.view(acc, (0, 0), (TILE_M, TILE_N), (MICRO_M, MICRO_N))
        sub_num = (K + SUB_BLK_K - 1) // SUB_BLK_K
        for k in tl.range(0, sub_num):
            a_descriptor_load = smt.descriptor_load(a_ptr, (0, 0))
            a = smt.view(
                a_descriptor_load,
                (0, k * SUB_BLK_K),
                (TILE_M, SUB_BLK_K),
                (MICRO_M, MICRO_K),
            )
            b_descriptor_load = smt.descriptor_load(b_ptr, (0, 0))
            b = smt.view(
                b_descriptor_load,
                (k * SUB_BLK_K, 0),
                (SUB_BLK_K, TILE_N),
                (MICRO_K, MICRO_N),
            )
            acc += smt.dot(a, b)
    acc = smt.view(acc, (0, 0), (TILE_M, TILE_N), (1, 1))

    c = acc.to(o_ptr.dtype.element_ty)

    tl.store(o_ptr, c, boundary_check=(0, 1))


def bmm(A, B):
    logger.debug("GEMS_SPACEMIT BMM")
    batch, M, K = A.shape
    _, _, N = B.shape
    if A.stride(0) > 1 and A.stride(1) > 1:
        A = A.contiguous()
    if B.stride(0) > 1 and B.stride(1) > 1:
        B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    if K == 1 and batch == 1:
        vec_a = A[0, :, 0]
        vec_b = B[0, 0, :]
        result = outer(vec_a, vec_b)
        return result.unsqueeze(0)

    if K == 1:
        return mul(A, B)

    def grid_fn(meta):
        return (
            triton.cdiv(meta["M"], meta["TILE_M"]),
            triton.cdiv(meta["N"], meta["TILE_N"]),
            batch,
        )

    TILE_K = triton.next_power_of_2(K)
    SUB_BLK_K = min(1024, TILE_K)

    bmm_kernel[grid_fn](
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
        TILE_K=TILE_K,
        SUB_BLK_K=SUB_BLK_K,
    )
    return out
