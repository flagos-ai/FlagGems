import logging
import os

import torch
import triton
import triton.language as tl
import yaml

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

from .utils import create_tma_device_descriptor, get_cached_tma_device_descriptor

logger = logging.getLogger("flag_gems.runtime.backend._mthreads.ops.mm")

EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_mthreads_expand.yaml")
)

SQMMA_SWITCH = False

def is_supported_sqmma_layout(tensor):
    return tensor.is_contiguous() or (
        tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
    )


def is_sqmma_compatible(a, b, N, K):
    return (
        os.getenv("MUSA_ENABLE_SQMMA", "0") == "1"
        and SQMMA_SWITCH
        and a.dim() == 2
        and b.dim() == 2
        and a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and is_supported_sqmma_layout(a)
        and is_supported_sqmma_layout(b)
        and N % 8 == 0
        and K % 8 == 0
    )


def get_expand_config(op):
    default_strategies = {
        "mm_sqmma": ["align32", "align32", "align32", "align32", "align32", "default"],
        "mm": ["align32", "align32", "align32", "align32", "align32"],
        "gemv": ["align32", "align32", "align32", "default"],
    }
    op_key_orders = {
        "mm_sqmma": ["M", "N", "K", "stride_am", "stride_bk", "dtype"],
        "mm": ["M", "N", "K", "stride_am", "stride_bk"],
        "gemv": ["M", "K", "stride_am", "stride_bk"],
    }
    op_meta_map = {
        "mm_sqmma": {
            "BM": "BLOCK_M",
            "BN": "BLOCK_N",
            "BK": "BLOCK_K",
        },
        "mm": {
            "BM": "BLOCK_M",
            "BN": "BLOCK_N",
            "BK": "BLOCK_K",
        },
        "gemv": {
            "BM": "BLOCK_M",
            "BK": "BLOCK_K",
        },
    }

    if op not in default_strategies:
        return -1

    default_strategy = default_strategies[op]
    if not os.path.exists(EXPAND_CONFIG_FILENAME):
        return -1

    try:
        with open(EXPAND_CONFIG_FILENAME, "r") as file:
            config = yaml.safe_load(file) or {}

        expand_configs = config.get(op)
        gen_config = None
        strategy_config = None
        for single_config in expand_configs:
            if isinstance(single_config, dict) and "param_map" in single_config:
                gen_config = single_config
            if isinstance(single_config, dict) and "strategy" in single_config:
                strategy_config = single_config.get("strategy")

        param_map = gen_config["param_map"]
        meta_map = param_map["META"]

        strategy = default_strategy
        if isinstance(strategy_config, dict):
            strategy = [
                strategy_config.get(k, default_strategy[idx])
                for idx, k in enumerate(op_key_orders[op])
            ]

        ranges = {}
        for range_key, meta_key in op_meta_map[op].items():
            ranges[range_key] = gen_config[meta_map[meta_key]]
        ranges["s"] = gen_config[param_map["num_stages"]]
        ranges["w"] = gen_config[param_map["num_warps"]]

        return {"ranges": ranges, "strategy": strategy}
    except Exception:
        return -1


def matmul_get_configs():
    if os.environ.get("USE_FLAGTUNE") == "1":
        expand_config = get_expand_config("mm")
        if expand_config != -1:
            ranges = expand_config["ranges"]
            return [
                triton.Config(
                    {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                    num_stages=s,
                    num_warps=w,
                )
                for BM in ranges["BM"]
                for BN in ranges["BN"]
                for BK in ranges["BK"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]
    return runtime.get_tuned_config("mm")


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=matmul_get_configs(),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=get_expand_config("mm")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1" and get_expand_config("mm") != -1
    else ["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=5,
)
@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    prev_multiple = prev_multiple_of(K, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # loop peeling
    rk = (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak), mask=mask_k[None, :]
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn), mask=mask_k[:, None]
    )
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


def gemv_get_configs():
    if os.environ.get("USE_FLAGTUNE") == "1":
        expand_config = get_expand_config("gemv")
        if expand_config != -1:
            ranges = expand_config["ranges"]
            return [
                triton.Config(
                    {"BLOCK_M": BM, "BLOCK_K": BK},
                    num_stages=s,
                    num_warps=w,
                )
                for BM in ranges["BM"]
                for BK in ranges["BK"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]
    return [triton.Config({"BLOCK_M": 64, "BLOCK_K": 64})]


@libentry()
@libtuner(
    configs=gemv_get_configs(),
    key=["M", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
)
@triton.jit
def gemv_kernel(
    A,
    B,
    C,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tle.program_id(0)

    row_start = pid * BLOCK_M
    row_offset = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offset < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offset = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        a_ptrs = A + row_offset[:, None] * stride_am + k_offset[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        b_ptrs = B + k_offset * stride_bk
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)

        # Keep the reduction in fp32 so N=1 GEMV matches the mm path more closely.
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.sum(a * b[None, :], axis=1)

    c_ptrs = C + row_offset * stride_cm
    acc = acc.to(C.dtype.element_ty)
    tl.store(c_ptrs, acc, mask=row_mask)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def mm_fma(a, b):
    logger.debug("GEMS_MTHREADS MM(FMA)")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dtype=str(a.dtype).split(".")[-1],
            GROUP_M=8,
        )
    return c


def gemv_mm(a, b, c, M, K):
    logger.debug(
        "GEMS_MTHREADS MM(GEMV), [shape info]: [%s, %s, 1](M, K, N)",
        M,
        K,
    )
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    with torch_device_fn.device(a.device):
        gemv_kernel[grid](
            a,
            b,
            c,
            M,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            c.stride(0),
        )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_MTHREADS MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = out
    if N == 1:
        return gemv_mm(a, b, c, M, K)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dtype=str(a.dtype).split(".")[-1],
            GROUP_M=8,
        )
    return c


def sqmma_descriptor_pre_hook(nargs):
    a = nargs["A"]
    b = nargs["B"]
    c = nargs["C"]
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    device = c.device

    nargs["a_desc_ptr"].copy_(
        get_cached_tma_device_descriptor(a, block_m, block_k, device)
    )
    nargs["b_desc_ptr"].copy_(
        get_cached_tma_device_descriptor(b, block_k, block_n, device)
    )
    nargs["c_desc_ptr"].copy_(create_tma_device_descriptor(c, block_m, block_n, device))


def sqmma_get_configs(pre_hook=sqmma_descriptor_pre_hook):
    if os.environ.get("USE_FLAGTUNE") == "1":
        expand_config = get_expand_config("mm_sqmma")
        if expand_config != -1:
            ranges = expand_config["ranges"]
            return [
                triton.Config(
                    {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for BM in ranges["BM"]
                for BN in ranges["BN"]
                for BK in ranges["BK"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=1,
            num_warps=4,
            pre_hook=pre_hook,
        )
    ]


@libentry()
@libtuner(
    configs=sqmma_get_configs(),
    key=["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    strategy=get_expand_config("mm_sqmma")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1" and get_expand_config("mm_sqmma") != -1
    else ["align32", "align32", "align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
)
@triton.jit
def mm_sqmma_kernel(
    A,
    B,
    C,
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    dtype: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ab_dtype: tl.constexpr,
    c_dtype: tl.constexpr,
    is_transpose_a: tl.constexpr = False,
    is_transpose_b: tl.constexpr = False,
):
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k = 0
    offs_am = offs_am.to(tl.int32)
    offs_bn = offs_bn.to(tl.int32)
    offs_k = offs_k.to(tl.int32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    tme_load_ab_dtype = ab_dtype
    c_store_dtype = c_dtype
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl._experimental_descriptor_load(
            a_desc_ptr,
            [offs_am, offs_k],
            [BLOCK_M, BLOCK_K],
            tme_load_ab_dtype,
            is_transpose_a,
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr,
            [offs_k, offs_bn],
            [BLOCK_K, BLOCK_N],
            tme_load_ab_dtype,
            is_transpose_b,
        )
        accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        offs_k += BLOCK_K
    accumulator = accumulator.to(c_store_dtype)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


def get_triton_type(elem_type):
    type_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float8_e4m3fn: tl.float8e4nv,
    }
    return type_map.get(elem_type, None)


def mm_sqmma(A, B, M, N, K, GROUP_M):
    logger.debug("GEMS_MTHREADS MM(SQMMA)")
    device = A.device
    # handle non-contiguous inputs if necessary
    is_transpose_a = False
    is_transpose_b = False
    if not A.is_contiguous():
        if A.stride(0) == 1 and A.stride(1) == A.shape[0]:
            is_transpose_a = True
        else:
            A = A.contiguous()
    if not B.is_contiguous():
        if B.stride(0) == 1 and B.stride(1) == B.shape[0]:
            is_transpose_b = True
        else:
            B = B.contiguous()
    a_type = A.dtype
    b_type = B.dtype
    assert a_type == b_type, "Mat A and Mat B should have the same dtype"
    c_dtype = get_higher_dtype(a_type, b_type)
    C = torch.empty((M, N), dtype=c_dtype, device=device)
    desc_a = torch.empty((64,), dtype=torch.int8, device=device)
    desc_b = torch.empty((64,), dtype=torch.int8, device=device)
    desc_c = torch.empty((64,), dtype=torch.int8, device=device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        1,
        1,
    )
    mm_sqmma_kernel[grid](
        A,
        B,
        C,
        desc_a,
        desc_b,
        desc_c,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        str(a_type).split(".")[-1],
        GROUP_M=GROUP_M,
        ab_dtype=get_triton_type(a_type),
        c_dtype=get_triton_type(c_dtype),
        is_transpose_a=is_transpose_a,
        is_transpose_b=is_transpose_b,
    )
    return C


def mm(a, b):
    a_dtype = a.dtype
    b_dtype = b.dtype
    M, K = a.shape
    _, N = b.shape
    if N == 1:
        c_dtype = get_higher_dtype(a_dtype, b_dtype)
        c = torch.empty((M, N), device=a.device, dtype=c_dtype)
        return gemv_mm(a, b, c, M, K)
    if is_sqmma_compatible(a, b, N, K):
        GROUP_M = 8
        return mm_sqmma(
            a,
            b,
            M,
            N,
            K,
            GROUP_M,
        )
    else:
        return mm_fma(a, b)
