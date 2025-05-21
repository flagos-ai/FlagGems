import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

from .utils import create_tma_device_descriptor, get_triton_dtype, should_enable_sqmma

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def bmm_sqmma_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    a_dtype: tl.constexpr,
    b_dtype: tl.constexpr,
    c_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # batch offsets
    pid_b = tle.program_id(2)

    pidx = tle.program_id(0)
    pidy = tle.program_id(1)

    pid_m, pid_n = pidx, pidy

    # add batch offset
    offs_m = pid_b * M
    offs_kb = pid_b * K

    offs_m = offs_m + pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    offs_k = 0
    offs_m = offs_m.to(tl.int32)
    offs_n = offs_n.to(tl.int32)
    offs_k = offs_k.to(tl.int32)
    offs_kb = offs_kb.to(tl.int32)
    atype = a_dtype
    btype = b_dtype
    num_iters = tl.cdiv(K, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(num_iters):
        a = tl._experimental_descriptor_load(
            A, [offs_m, offs_k], [BLOCK_M, BLOCK_K], atype
        )
        b = tl._experimental_descriptor_load(
            B, [offs_kb, offs_n], [BLOCK_K, BLOCK_N], btype
        )

        acc += tl.dot(a, b, allow_tf32=False)
        offs_kb += BLOCK_K
        offs_k += BLOCK_K

    acc = acc.to(c_dtype)

    tl._experimental_descriptor_store(O, acc, [offs_m, offs_n])


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("bmm"),
    key=["M", "N", "K"],
)
@triton.heuristics(runtime.get_heuristic_config("bmm"))
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    # batch offsets
    pid_b = tle.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = tle.program_id(0)
    pidy = tle.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = tle.num_programs(0)
        gridy = tle.num_programs(1)
        pid = pidx + pidy * gridx

        num_CTA_per_group = gridy * GROUP_M

        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask_a)
        b = tl.load(b_ptrs, mask_b)

        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        o += tl.dot(a, b, allow_tf32=False)

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    elif DIVISIBLE_M and not DIVISIBLE_N:
        mask_c = mask_n[None, :]
    elif not DIVISIBLE_M and DIVISIBLE_N:
        mask_c = mask_m[:, None]
    else:
        mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)


def get_mm_config():
    return {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 1,
        "num_stages": 1,
        "num_warps": 4,
    }


def bmm_sqmma(A, B):
    logger.debug("GEMS BMM SQMMA")
    device = A.device
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    # allocates output
    c_dtype = A.dtype if A.dtype != torch.bfloat16 else torch.float32
    c = torch.empty((batch, M, N), device=device, dtype=c_dtype)
    a_dtype = get_triton_dtype(A.dtype)
    b_dtype = get_triton_dtype(B.dtype)
    c_dtype = get_triton_dtype(c_dtype)
    # prepare tma descriptor for sqmma
    mm_config = get_mm_config()
    BLOCK_M = mm_config["BLOCK_M"]
    BLOCK_N = mm_config["BLOCK_N"]
    BLOCK_K = mm_config["BLOCK_K"]
    GROUP_M = mm_config["GROUP_M"]
    num_stages = mm_config["num_stages"]
    num_warps = mm_config["num_warps"]
    desc_a = create_tma_device_descriptor(
        A.reshape(batch * M, K), BLOCK_M, BLOCK_K, device
    )
    desc_b = create_tma_device_descriptor(
        B.reshape(batch * K, N), BLOCK_K, BLOCK_N, device
    )
    desc_c = create_tma_device_descriptor(
        c.reshape(batch * M, N), BLOCK_M, BLOCK_N, device
    )

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["BLOCK_M"]),
        triton.cdiv(meta["N"], meta["BLOCK_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_sqmma_kernel[grid_fn](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            c_dtype=c_dtype,
            GROUP_M=GROUP_M,
            num_stages=num_stages,
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

    return c.to(A.dtype)


def bmm_fma(A, B):
    logger.debug("GEMS BMM FMA")
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_kernel[grid_fn](A, B, out, M, N, K)
    return out


def bmm(a, b):
    a_dtype = a.dtype
    b_dtype = b.dtype
    batch, M, K = a.shape
    _, _, N = b.shape
    use_sqmma = should_enable_sqmma(a_dtype, b_dtype, M, N, K)
    if use_sqmma:
        return bmm_sqmma(a, b)
    else:
        enable_sqmma = os.environ.pop("MUSA_ENABLE_SQMMA", None)
        result = bmm_fma(a, b)
        if enable_sqmma:
            os.environ["MUSA_ENABLE_SQMMA"] = enable_sqmma
        return result
