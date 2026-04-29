import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

from .utils import create_tma_device_descriptor, get_cached_tma_device_descriptor

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "bmm_mthreads_expand.yaml")
)


def is_supported_sqmma_layout(tensor):
    return tensor.is_contiguous() or (
        tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
    )


def is_sqmma_compatible(a, b, N, K):
    return (
        a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and is_supported_sqmma_layout(a)
        and is_supported_sqmma_layout(b)
        and N % 8 == 0
        and K % 8 == 0
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
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
    IS_FP64: tl.constexpr = False,
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


def bmm_fma(A, B):
    logger.debug("GEMS_MTHREADS BMM(FMA)")
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
        bmm_kernel[grid_fn](A, B, out, M, N, K, IS_FP64=A.dtype == torch.float64)
    return out


def bmm_sqmma_descriptor_pre_hook(nargs):
    a = nargs["A"]
    b = nargs["B"]
    c = nargs["C"]
    batch = nargs["batch"]
    M = nargs["M"]
    N = nargs["N"]
    K = nargs["K"]
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    device = c.device

    nargs["a_desc_ptr"].copy_(
        get_cached_tma_device_descriptor(
            a.reshape(batch * M, K), block_m, block_k, device
        )
    )
    nargs["b_desc_ptr"].copy_(
        get_cached_tma_device_descriptor(
            b.reshape(batch * K, N), block_k, block_n, device
        )
    )
    nargs["c_desc_ptr"].copy_(
        create_tma_device_descriptor(c.reshape(batch * M, N), block_m, block_n, device)
    )


@libentry()
@libtuner(
    configs=runtime.ops_get_configs(
        "bmm_sqmma",
        pre_hook=bmm_sqmma_descriptor_pre_hook,
        yaml_path=EXPAND_CONFIG_FILENAME,
    )
    if os.environ.get("USE_FLAGTUNE") == "1"
    else [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=1,
            num_warps=4,
            pre_hook=bmm_sqmma_descriptor_pre_hook,
        )
    ],
    key=["M", "N", "K"],
    strategy=runtime.get_expand_config("bmm_sqmma", yaml_path=EXPAND_CONFIG_FILENAME)[
        "strategy"
    ][:3]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32"],
    warmup=5,
    rep=5,
)
@triton.jit
def bmm_sqmma_kernel(
    A,
    B,
    C,
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    batch,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ab_type: tl.constexpr,
    d_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_index = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M + batch_index * M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_ak = 0
    offs_bk = batch_index * K
    tme_load_type = ab_type
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_ak], [BLOCK_SIZE_M, BLOCK_SIZE_K], tme_load_type
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bk, offs_bn], [BLOCK_SIZE_K, BLOCK_SIZE_N], tme_load_type
        )
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_ak += BLOCK_SIZE_K
        offs_bk += BLOCK_SIZE_K
    accumulator = accumulator.to(d_type)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


def get_triton_type(elem_type):
    type_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float8_e4m3fn: tl.float8e4nv,
    }
    return type_map.get(elem_type, None)


def bmm_sqmma(A, B, elem_type, batch, M, N, K):
    device = "musa"
    ab_type = elem_type
    c_type = elem_type if (elem_type != torch.bfloat16) else torch.float16
    C = torch.empty((batch, M, N), dtype=torch.float16, device=device).to(c_type)
    desc_a = torch.empty((64,), dtype=torch.int8, device=device)
    desc_b = torch.empty((64,), dtype=torch.int8, device=device)
    desc_c = torch.empty((64,), dtype=torch.int8, device=device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        batch,
        1,
    )
    bmm_sqmma_kernel[grid](
        A,
        B,
        C,
        desc_a,
        desc_b,
        desc_c,
        batch,
        M,
        N,
        K,
        ab_type=get_triton_type(ab_type),
        d_type=get_triton_type(c_type),
    )
    return C


def bmm(a, b):
    a_dtype = a.dtype
    b_dtype = b.dtype
    batch, M, K = a.shape
    _, _, N = b.shape
    need_sqmma = a_dtype != torch.float32 and b_dtype != torch.float32
    prev_sqmma = os.environ.get("MUSA_ENABLE_SQMMA")
    if need_sqmma:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
    else:
        os.environ.pop("MUSA_ENABLE_SQMMA", None)
    try:
        if is_sqmma_compatible(a, b, N, K):
            return bmm_sqmma(a, b, a_dtype, batch, M, N, K)
        else:
            return bmm_fma(a, b)
    finally:
        if prev_sqmma is None:
            os.environ.pop("MUSA_ENABLE_SQMMA", None)
        else:
            os.environ["MUSA_ENABLE_SQMMA"] = prev_sqmma
