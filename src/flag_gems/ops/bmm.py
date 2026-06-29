import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=[
        "log",
        "log",
        "log",
        "align32",
        "align32",
    ],
    flagtune_op_name="bmm",
    flagtune_expand_op_name="bmm",
    flagtune_pre_hook=None,
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
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        # reorder CTAs
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
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

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    num_iters = tl.cdiv(K, TILE_K)
    if IS_FP64:
        o = tl.zeros((TILE_M, TILE_N), dtype=tl.float64)
    else:
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
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

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


def bmm(A, B):
    logger.debug("GEMS BMM")
    assert A.shape[0] == B.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape
    out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
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
            IS_FP64=A.dtype == torch.float64,
        )
    return out


def bmm_out(A, B, out):
    logger.debug("GEMS BMM_OUT")
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape

    grid_fn = lambda meta: (
        triton.cdiv(meta["M"], meta["TILE_M"]),
        triton.cdiv(meta["N"], meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
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
            IS_FP64=A.dtype == torch.float64,
        )
    return out


@libentry()
@triton.jit
def bmm_fp8_w8a16_kernel(
    A,
    B,
    B_SCALE,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_skb: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_b = ext.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    B_SCALE += pid_b * stride_sb
    O += pid_b * stride_ob

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx
        num_cta_per_group = gridy * GROUP_M
        group_id = pid // num_cta_per_group
        inner_group_id = pid % num_cta_per_group
        group_size = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % group_size
        pid_n = inner_group_id // group_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    scale_ptrs = B_SCALE + offs_n * stride_sn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k_tile in range(NUM_K_TILES):
        k_offsets = k_tile * TILE_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_q = tl.load(b_ptrs, mask=b_mask, other=0.0)
        scale_block_idx = (k_tile * TILE_K) // BLOCK_SIZE
        b_scale = tl.load(
            scale_ptrs + scale_block_idx * stride_skb,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.bfloat16)
        b = b_q.to(tl.bfloat16) * b_scale[None, :]
        acc += tl.dot(a, b, allow_tf32=False)

        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=o_mask)


def bmm_fp8_w8a16(A, B, B_scale, block_size=128, out_dtype=torch.bfloat16):
    logger.debug("GEMS BMM FP8 W8A16")
    assert A.ndim == 3 and B.ndim == 3 and B_scale.ndim == 3
    assert A.shape[0] == B.shape[0] == B_scale.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    batch, M, K = A.shape
    _, _, N = B.shape
    num_k_blocks = triton.cdiv(K, block_size)
    assert B_scale.shape == (
        batch,
        num_k_blocks,
        N,
    ), f"B_scale shape should be {(batch, num_k_blocks, N)}, got {B_scale.shape}"
    assert A.dtype in (torch.float16, torch.bfloat16), "A must be fp16 or bf16"
    assert B.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ), "B must be fp8"

    out = torch.empty((batch, M, N), dtype=out_dtype, device=A.device)

    def grid_fn(meta):
        return (
            triton.cdiv(M, meta["TILE_M"]),
            triton.cdiv(N, meta["TILE_N"]),
            batch,
        )

    if M <= 32 and N <= 64:
        tile_m, tile_n, tile_k, group_m, num_warps, num_stages = 64, 16, 64, 4, 4, 3
    elif M <= 128 and N <= 128:
        tile_m, tile_n, tile_k, group_m, num_warps, num_stages = 16, 32, 64, 8, 4, 2
    elif M >= 512 and N >= 512:
        tile_m, tile_n, tile_k, group_m, num_warps, num_stages = 64, 64, 32, 4, 4, 3
    else:
        tile_m, tile_n, tile_k, group_m, num_warps, num_stages = 64, 16, 64, 4, 4, 4
    assert block_size % tile_k == 0, "block_size must be divisible by TILE_K"

    with torch_device_fn.device(A.device):
        bmm_fp8_w8a16_kernel[grid_fn](
            A,
            B,
            B_scale,
            out,
            M,
            N,
            K,
            NUM_K_TILES=triton.cdiv(K, tile_k),
            BLOCK_SIZE=block_size,
            stride_ab=A.stride(0),
            stride_am=A.stride(1),
            stride_ak=A.stride(2),
            stride_bb=B.stride(0),
            stride_bk=B.stride(1),
            stride_bn=B.stride(2),
            stride_sb=B_scale.stride(0),
            stride_skb=B_scale.stride(1),
            stride_sn=B_scale.stride(2),
            stride_ob=out.stride(0),
            stride_om=out.stride(1),
            stride_on=out.stride(2),
            TILE_M=tile_m,
            TILE_N=tile_n,
            TILE_K=tile_k,
            GROUP_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


@libentry()
@triton.jit
def bmm_fp8_w8a8_kernel(
    A,
    B,
    A_SCALE,
    B_SCALE,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_asb: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsb: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_b = ext.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    A_SCALE += pid_b * stride_asb
    B_SCALE += pid_b * stride_bsb
    O += pid_b * stride_ob

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx
        num_cta_per_group = gridy * GROUP_M
        group_id = pid // num_cta_per_group
        inner_group_id = pid % num_cta_per_group
        group_size = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % group_size
        pid_n = inner_group_id // group_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = A_SCALE + offs_m * stride_asm
    n_block_idx = (pid_n * TILE_N) // BLOCK_N
    b_scale_ptr = B_SCALE + n_block_idx * stride_bsn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    if BLOCK_K == K:
        a_scale = tl.load(
            a_scale_ptrs,
            mask=offs_m < M,
            other=0.0,
        ).to(tl.float32)
        b_scale = tl.load(b_scale_ptr).to(tl.float32)
        for k_tile in range(NUM_K_TILES):
            k_offsets = k_tile * TILE_K + offs_k
            a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
            b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)

            a_ptrs += TILE_K * stride_ak
            b_ptrs += TILE_K * stride_bk
        acc = acc * a_scale[:, None] * b_scale
    else:
        for k_tile in range(NUM_K_TILES):
            k_offsets = k_tile * TILE_K + offs_k
            k_block_idx = (k_tile * TILE_K) // BLOCK_K
            a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
            b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            partial = tl.dot(a, b, allow_tf32=False)

            a_scale = tl.load(
                a_scale_ptrs + k_block_idx * stride_ask,
                mask=offs_m < M,
                other=0.0,
            ).to(tl.float32)
            b_scale = tl.load(b_scale_ptr + k_block_idx * stride_bsk).to(tl.float32)
            acc += partial * a_scale[:, None] * b_scale

            a_ptrs += TILE_K * stride_ak
            b_ptrs += TILE_K * stride_bk

    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=o_mask)


@libentry()
@triton.jit
def bmm_fp8_w8a8_packed_scale_kernel(
    A,
    B,
    SCALE,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sk: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_b = ext.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    SCALE += pid_b * stride_sb
    O += pid_b * stride_ob

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx
        num_cta_per_group = gridy * GROUP_M
        group_id = pid // num_cta_per_group
        inner_group_id = pid % num_cta_per_group
        group_size = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % group_size
        pid_n = inner_group_id // group_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    n_block_idx = (pid_n * TILE_N) // BLOCK_N
    scale_ptrs = SCALE + offs_m * stride_sm + n_block_idx * stride_sn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k_tile in range(NUM_K_TILES):
        k_offsets = k_tile * TILE_K + offs_k
        k_block_idx = (k_tile * TILE_K) // BLOCK_K
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        partial = tl.dot(a, b, allow_tf32=False)
        scale = tl.load(
            scale_ptrs + k_block_idx * stride_sk,
            mask=offs_m < M,
            other=0.0,
        ).to(tl.float32)
        acc += partial * scale[:, None]

        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=o_mask)


@libentry()
@triton.jit
def bmm_fp8_w8a8_block_scale_kernel(
    A,
    B,
    A_SCALE,
    B_SCALE,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_asb: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bsb: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_b = ext.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    A_SCALE += pid_b * stride_asb
    B_SCALE += pid_b * stride_bsb
    O += pid_b * stride_ob

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx
        num_cta_per_group = gridy * GROUP_M
        group_id = pid // num_cta_per_group
        inner_group_id = pid % num_cta_per_group
        group_size = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % group_size
        pid_n = inner_group_id // group_size

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    m_block_idx = (pid_m * TILE_M) // BLOCK_M
    n_block_idx = (pid_n * TILE_N) // BLOCK_N
    a_scale_ptr = A_SCALE + m_block_idx * stride_asm
    b_scale_ptr = B_SCALE + n_block_idx * stride_bsn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    if BLOCK_K == K:
        a_scale = tl.load(a_scale_ptr).to(tl.float32)
        b_scale = tl.load(b_scale_ptr).to(tl.float32)
        for k_tile in range(NUM_K_TILES):
            k_offsets = k_tile * TILE_K + offs_k
            a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
            b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b, allow_tf32=False)
            a_ptrs += TILE_K * stride_ak
            b_ptrs += TILE_K * stride_bk
        acc = acc * a_scale * b_scale
    else:
        for k_tile in range(NUM_K_TILES):
            k_offsets = k_tile * TILE_K + offs_k
            k_block_idx = (k_tile * TILE_K) // BLOCK_K
            a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
            b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            partial = tl.dot(a, b, allow_tf32=False)
            a_scale = tl.load(a_scale_ptr + k_block_idx * stride_ask).to(tl.float32)
            b_scale = tl.load(b_scale_ptr + k_block_idx * stride_bsk).to(tl.float32)
            acc += partial * a_scale * b_scale
            a_ptrs += TILE_K * stride_ak
            b_ptrs += TILE_K * stride_bk

    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptrs, acc, mask=o_mask)


def _get_bmm_fp8_w8a8_config(M, N, K, block_k):
    if K < block_k:
        return 64, 16, 64, 2, 4, 2
    if M <= 64 and N <= 64:
        return 128, 32, 128, 2, 8, 3
    if M <= 128 and N <= 128:
        return 32, 16, 128, 2, 4, 2
    if M <= 256 and N <= 256:
        return 64, 32, 128, 8, 4, 2
    if M >= 512 and N >= 512:
        return 128, 32, 128, 2, 4, 2
    return 64, 16, 128, 2, 4, 2


def _get_bmm_fp8_w8a8_block_scale_config(M, N, K, block_k):
    if K < block_k:
        return 64, 16, 64, 2, 4, 2
    if M <= 64 and N <= 64:
        return 128, 32, 128, 2, 8, 3
    if M <= 128 and N <= 128:
        return 32, 16, 128, 2, 4, 2
    if M >= 512 and N >= 512:
        return 128, 32, 128, 2, 4, 2
    return 64, 16, 128, 2, 4, 2


def bmm_fp8_w8a8(
    A,
    B,
    A_scale,
    B_scale,
    block_size=(128, 128),
    out_dtype=torch.bfloat16,
    out=None,
):
    logger.debug("GEMS BMM FP8 W8A8")
    assert A.ndim == 3 and B.ndim == 3 and A_scale.ndim == 3 and B_scale.ndim == 3
    assert A.shape[0] == B.shape[0] == A_scale.shape[0] == B_scale.shape[0]
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    assert A.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "A must be fp8"
    assert B.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "B must be fp8"

    block_n, block_k = block_size
    batch, M, K = A.shape
    _, _, N = B.shape
    num_k_blocks = triton.cdiv(K, block_k)
    num_n_blocks = triton.cdiv(N, block_n)
    assert A_scale.shape == (batch, M, num_k_blocks)
    assert B_scale.shape == (batch, num_k_blocks, num_n_blocks)

    if out is None:
        out = torch.empty((batch, M, N), dtype=out_dtype, device=A.device)
    else:
        assert out.shape == (batch, M, N)
        assert out.dtype == out_dtype and out.device == A.device

    def grid_fn(meta):
        return (
            triton.cdiv(M, meta["TILE_M"]),
            triton.cdiv(N, meta["TILE_N"]),
            batch,
        )

    tile_m, tile_n, tile_k, group_m, num_warps, num_stages = _get_bmm_fp8_w8a8_config(
        M, N, K, block_k
    )
    assert block_k % tile_k == 0, "block_k must be divisible by TILE_K"
    assert block_n % tile_n == 0, "block_n must be divisible by TILE_N"

    with torch_device_fn.device(A.device):
        bmm_fp8_w8a8_kernel[grid_fn](
            A,
            B,
            A_scale,
            B_scale,
            out,
            M,
            N,
            K,
            NUM_K_TILES=triton.cdiv(K, tile_k),
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            stride_ab=A.stride(0),
            stride_am=A.stride(1),
            stride_ak=A.stride(2),
            stride_bb=B.stride(0),
            stride_bk=B.stride(1),
            stride_bn=B.stride(2),
            stride_asb=A_scale.stride(0),
            stride_asm=A_scale.stride(1),
            stride_ask=A_scale.stride(2),
            stride_bsb=B_scale.stride(0),
            stride_bsk=B_scale.stride(1),
            stride_bsn=B_scale.stride(2),
            stride_ob=out.stride(0),
            stride_om=out.stride(1),
            stride_on=out.stride(2),
            TILE_M=tile_m,
            TILE_N=tile_n,
            TILE_K=tile_k,
            GROUP_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def bmm_fp8_w8a8_packed_scale(
    A,
    B,
    scale,
    block_size=(128, 128),
    out_dtype=torch.bfloat16,
    out=None,
):
    logger.debug("GEMS BMM FP8 W8A8 PACKED_SCALE")
    assert A.ndim == 3 and B.ndim == 3 and scale.ndim == 4
    assert A.shape[0] == B.shape[0] == scale.shape[0]
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    assert A.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "A must be fp8"
    assert B.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "B must be fp8"

    block_n, block_k = block_size
    batch, M, K = A.shape
    _, _, N = B.shape
    num_k_blocks = triton.cdiv(K, block_k)
    num_n_blocks = triton.cdiv(N, block_n)
    assert scale.shape == (batch, M, num_k_blocks, num_n_blocks)

    if out is None:
        out = torch.empty((batch, M, N), dtype=out_dtype, device=A.device)
    else:
        assert out.shape == (batch, M, N)
        assert out.dtype == out_dtype and out.device == A.device

    def grid_fn(meta):
        return (
            triton.cdiv(M, meta["TILE_M"]),
            triton.cdiv(N, meta["TILE_N"]),
            batch,
        )

    tile_m, tile_n, tile_k, group_m, num_warps, num_stages = _get_bmm_fp8_w8a8_config(
        M, N, K, block_k
    )
    assert block_k % tile_k == 0, "block_k must be divisible by TILE_K"
    assert block_n % tile_n == 0, "block_n must be divisible by TILE_N"

    with torch_device_fn.device(A.device):
        bmm_fp8_w8a8_packed_scale_kernel[grid_fn](
            A,
            B,
            scale,
            out,
            M,
            N,
            K,
            NUM_K_TILES=triton.cdiv(K, tile_k),
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            stride_ab=A.stride(0),
            stride_am=A.stride(1),
            stride_ak=A.stride(2),
            stride_bb=B.stride(0),
            stride_bk=B.stride(1),
            stride_bn=B.stride(2),
            stride_sb=scale.stride(0),
            stride_sm=scale.stride(1),
            stride_sk=scale.stride(2),
            stride_sn=scale.stride(3),
            stride_ob=out.stride(0),
            stride_om=out.stride(1),
            stride_on=out.stride(2),
            TILE_M=tile_m,
            TILE_N=tile_n,
            TILE_K=tile_k,
            GROUP_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def bmm_fp8_w8a8_block_scale(
    A,
    B,
    A_scale,
    B_scale,
    block_size=(128, 128, 128),
    out_dtype=torch.bfloat16,
    out=None,
):
    logger.debug("GEMS BMM FP8 W8A8 BLOCK_SCALE")
    assert A.ndim == 3 and B.ndim == 3 and A_scale.ndim == 3 and B_scale.ndim == 3
    assert A.shape[0] == B.shape[0] == A_scale.shape[0] == B_scale.shape[0]
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    assert A.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "A must be fp8"
    assert B.dtype in (torch.float8_e4m3fn, torch.float8_e5m2), "B must be fp8"

    block_m, block_n, block_k = block_size
    batch, M, K = A.shape
    _, _, N = B.shape
    num_m_blocks = triton.cdiv(M, block_m)
    num_k_blocks = triton.cdiv(K, block_k)
    num_n_blocks = triton.cdiv(N, block_n)
    assert A_scale.shape == (batch, num_m_blocks, num_k_blocks)
    assert B_scale.shape == (batch, num_k_blocks, num_n_blocks)

    if out is None:
        out = torch.empty((batch, M, N), dtype=out_dtype, device=A.device)
    else:
        assert out.shape == (batch, M, N)
        assert out.dtype == out_dtype and out.device == A.device

    def grid_fn(meta):
        return (
            triton.cdiv(M, meta["TILE_M"]),
            triton.cdiv(N, meta["TILE_N"]),
            batch,
        )

    (
        tile_m,
        tile_n,
        tile_k,
        group_m,
        num_warps,
        num_stages,
    ) = _get_bmm_fp8_w8a8_block_scale_config(M, N, K, block_k)
    assert block_m % tile_m == 0 or tile_m % block_m == 0
    assert block_k % tile_k == 0, "block_k must be divisible by TILE_K"
    assert block_n % tile_n == 0, "block_n must be divisible by TILE_N"

    with torch_device_fn.device(A.device):
        bmm_fp8_w8a8_block_scale_kernel[grid_fn](
            A,
            B,
            A_scale,
            B_scale,
            out,
            M,
            N,
            K,
            NUM_K_TILES=triton.cdiv(K, tile_k),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            stride_ab=A.stride(0),
            stride_am=A.stride(1),
            stride_ak=A.stride(2),
            stride_bb=B.stride(0),
            stride_bk=B.stride(1),
            stride_bn=B.stride(2),
            stride_asb=A_scale.stride(0),
            stride_asm=A_scale.stride(1),
            stride_ask=A_scale.stride(2),
            stride_bsb=B_scale.stride(0),
            stride_bsk=B_scale.stride(1),
            stride_bsn=B_scale.stride(2),
            stride_ob=out.stride(0),
            stride_om=out.stride(1),
            stride_on=out.stride(2),
            TILE_M=tile_m,
            TILE_N=tile_n,
            TILE_K=tile_k,
            GROUP_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
