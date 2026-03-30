import logging
import os
from typing import Optional
import torch
import triton
import triton.language as tl
import yaml

from flag_gems import runtime
from flag_gems.ops.mm_streamk import streamk_mm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.device_info import get_device_capability, get_sm_count

logger = logging.getLogger("flag_gems.runtime.backend._nvidia.hopper.ops.mm")
CACHE_USAGE_THRESHOLD = 0.8


def is_tma_compatible(a, b, N, K):
    """
    Check if tensors are compatible with TMA (Tensor Memory Accelerator).

    TMA requires 128-bit (16-byte) alignment for memory access:
    - For FP16/BF16 (2 bytes/element): N and K must be multiples of 8
      (8 elements × 2 bytes = 16 bytes)
    - For FP32 (4 bytes/element): N and K must be multiples of 4
      (4 elements × 4 bytes = 16 bytes)

    Args:
        a, b: Input tensors
        N, K: Matrix dimensions

    Returns:
        bool: True if compatible with TMA's 128-bit alignment requirement
    """
    return (
        a.dtype in (torch.float16, torch.bfloat16)
        and b.dtype in (torch.float16, torch.bfloat16)
        and N % 8 == 0
        and K % 8 == 0
    ) or (
        a.dtype in (torch.float32,)
        and b.dtype in (torch.float32,)
        and N % 4 == 0
        and K % 4 == 0
    )


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    if nargs["A_ROW_MAJOR"]:
        nargs["a_desc"].block_shape = [BLOCK_M, BLOCK_K]
    else:
        nargs["a_desc"].block_shape = [BLOCK_K, BLOCK_M]

    if nargs["B_ROW_MAJOR"]:
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    else:
        nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]

    nargs["c_desc"].block_shape = [BLOCK_M, BLOCK_N]


def get_expand_config(op):
    default_strategies = {
        "matmul": ["align32", "align32", "align32", "align32", "align32", "default"],
        "gemv": ["align32", "align32", "align32", "default"],
    }
    op_key_orders = {
        "matmul": ["M", "N", "K", "stride_am", "stride_bk", "dtype"],
        "gemv": ["M", "K", "stride_am", "stride_bk"],
    }
    op_meta_map = {
        "matmul": {
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
    config_path = os.path.join(os.path.dirname(__file__), "..", "mm_hopper_expand.yaml")
    if not os.path.exists(config_path):
        return -1

    try:
        with open(config_path, "r") as file:
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

        return {
            "ranges": ranges,
            "strategy": strategy,
        }
    except Exception:
        return -1


def matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook):
    if os.environ.get("USE_FLAGTUNE") == "1":
        expand_config = get_expand_config("matmul")
        if expand_config != -1:
            logger.debug(
                "Using expand configurations from mm_hopper_expand.yaml for matmul kernel autotuning"
            )
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
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in [32, 64, 128, 256]
        for BN in [32, 64, 128]
        for BK in [32, 64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("mm_general", pre_hook=None)
    if os.environ.get("USE_FLAGTUNE") == "1"
    else runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=runtime.get_expand_config("mm_general")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["default", "default", "default", "default", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def mm_kernel_general(
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

    if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0:
        # offset
        offset_am = pid_m * BLOCK_M
        offset_bn = pid_n * BLOCK_N
        offset_k = 0

        a_desc = tl.make_tensor_descriptor(
            base=A,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )

        # row-major
        b_desc = tl.make_tensor_descriptor(
            base=B,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_K, BLOCK_N],
        )

        # column-major
        # b_desc = tl.make_tensor_descriptor(
        #     B,
        #     shape = [N, K],
        #     strides = [K, 1],
        #     block_shape = [BLOCK_N, BLOCK_K],
        # )

        c_desc = tl.make_tensor_descriptor(
            base=C,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([offset_am.to(tl.int32), offset_k.to(tl.int32)])
            b = b_desc.load([offset_k.to(tl.int32), offset_bn.to(tl.int32)])
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
            offset_k += BLOCK_K

        acc = acc.to(a_desc.dtype)
        c_desc.store([offset_am.to(tl.int32), offset_bn.to(tl.int32)], acc)

    else:
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
            A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn),
            mask=mask_k[:, None],
            other=0.0,
        )
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

        acc = acc.to(C.dtype.element_ty)
        # rematerialize rm and rn to save registers
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        offsets = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # handles write-back with reduction-splitting
        tl.store(offsets, acc, mask=mask)


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("mm_general_tma", pre_hook=matmul_tma_set_block_size_hook)
    if os.environ.get("USE_FLAGTUNE") == "1"
    else [
        triton.Config(
            dict(config.kwargs),
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            num_ctas=getattr(config, "num_ctas", 1),
            pre_hook=matmul_tma_set_block_size_hook,
        )
        for config in runtime.get_tuned_config("mm")
    ],
    key=["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    strategy=runtime.get_expand_config("mm_general_tma")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
)
@triton.jit
def mm_kernel_general_host_tma(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
    dtype: tl.constexpr,
    enable_warp_specialization=True,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offset_am = (pid_m * BLOCK_M).to(tl.int32)
    offset_bn = (pid_n * BLOCK_N).to(tl.int32)
    iters = tl.cdiv(K, BLOCK_K)
    for k in range(iters):
        offset_ak = (k * BLOCK_K).to(tl.int32)

        if A_ROW_MAJOR:
            a = a_desc.load([offset_am, offset_ak])
        else:
            a_t = a_desc.load([offset_ak, offset_am])
            a = tl.trans(a_t)

        if B_ROW_MAJOR:
            b = b_desc.load([offset_ak, offset_bn])
        else:
            b_t = b_desc.load([offset_bn, offset_ak])
            b = tl.trans(b_t)

        if a_desc.dtype == tl.float16 or a_desc.dtype == tl.bfloat16:
            accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)
        else:
            accumulator = tl.dot(a, b, acc=accumulator, input_precision="tf32x3")

    c = accumulator.to(c_desc.dtype)
    c_desc.store([offset_am, offset_bn], c)


def get_higher_dtype(a, b):
    _ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]

    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def general_mm(a, b, c, M, N, K):
    # TODO: Remove this debug message
    logger.debug(
        "GEMS MM-hopper, [mm scenario]: general, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    if hasattr(
        triton.tools.tensor_descriptor, "TensorDescriptor"
    ) and is_tma_compatible(a, b, N, K):
        a_row_major = a.stride(1) == 1
        b_row_major = b.stride(1) == 1
        dummy_block = [1, 1]
        # triton 3.5.0
        from triton.tools.tensor_descriptor import TensorDescriptor

        if a_row_major:
            a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
        else:
            a_desc = TensorDescriptor(a, a.T.shape, a.T.stride(), dummy_block)
        if b_row_major:
            b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
        else:
            b_desc = TensorDescriptor(b, b.T.shape, b.T.stride(), dummy_block)
        c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

        input_dtype = a.dtype
        dtype_str = str(input_dtype).split(".")[-1]

        with torch_device_fn.device(a.device):
            mm_kernel_general_host_tma[grid](
                a_desc,
                b_desc,
                c_desc,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                GROUP_M=8,
                A_ROW_MAJOR=a_row_major,
                B_ROW_MAJOR=b_row_major,
                dtype=dtype_str,
            )
    else:

        def alloc_fn(size: int, align: int, stream: Optional[int]):
            return torch.empty(size, dtype=torch.int8, device=a.device)

        triton.set_allocator(alloc_fn)

        with torch_device_fn.device(a.device):
            mm_kernel_general[grid](
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
                GROUP_M=8,
            )
    return c


def gemv_get_configs():
    if os.environ.get("USE_FLAGTUNE") == "1":
        expand_config = get_expand_config("gemv")
        if expand_config != -1:
            logger.debug(
                "Using expand configurations from mm_hopper_expand.yaml for gemv kernel autotuning"
            )
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
    return [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_K": 256},
        )
    ]


@libentry()
@libtuner(
    configs=runtime.ops_get_configs("mm_gemv",pre_hook=None)
    if os.environ.get("USE_FLAGTUNE") == "1"
    else [triton.Config({"BLOCK_M": 32, "BLOCK_K": 256})],
    key=["M", "K", "stride_am", "stride_bk"],
    strategy=runtime.get_expand_config("mm_gemv") ["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32", "default"],
    warmup=5,
    rep=10,
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
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Optimized kernel for matrix-vector multiplication (N=1 case)"""
    pid = tl.program_id(0)

    # Each program handles BLOCK_M rows
    row_start = pid * BLOCK_M
    row_offset = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offset < M

    # Accumulator for this block of rows
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offset = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        # Load block from matrix A: [BLOCK_M, BLOCK_K]
        a_ptrs = A + row_offset[:, None] * stride_am + k_offset[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Load block from vector B: [BLOCK_K]
        b_ptrs = B + k_offset * stride_bk
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)

        # Accumulate: sum over K dimension
        acc += tl.sum(a * b[None, :], axis=1)

    # Store result
    c_ptrs = C + row_offset
    acc = acc.to(C.dtype.element_ty)
    tl.store(c_ptrs, acc, mask=row_mask)


def gemv_mm(a, b, c, M, K):
    """Optimized matrix-vector multiplication for N=1 case"""
    logger.debug(
        "GEMS MM-hopper, [mm scenario]: gemv (N=1), [shape info]: [%s, %s, 1](M, K, N)",
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
        )
    return c


def streamk_scenario(a, b, M, N, K):
    # TODO: this my change sometime according to the realbenchmark result
    # Currently, the best configuration for streamk has only been tested on A100(capability[0] == 8).
    # The optimal settings for other devices need to be determined through real testing.
    capability = get_device_capability()
    return (
        capability[0] == 8
        and a.dtype in [torch.float16, torch.bfloat16]
        and b.dtype in [torch.float16, torch.bfloat16]
        and a.is_contiguous()
        and b.is_contiguous()
        and K > M * 5
        and K > N * 5
    )


def mm(a, b):
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

    # Optimize for N=1 case (matrix-vector multiplication)
    if N == 1:
        return gemv_mm(a, b, c, M, K)
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, c, M, N, K, sm_count=sm_count)
    else:
        return general_mm(a, b, c, M, N, K)


def mm_out(a, b, *, out):
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    # Optimize for N=1 case (matrix-vector multiplication)
    if N == 1:
        return gemv_mm(a, b, out, M, K)
    # l2_cache_size = get_l2_cache_size()
    sm_count = get_sm_count()
    if streamk_scenario(a, b, M, N, K):
        return streamk_mm(a, b, out, M, N, K, sm_count=sm_count)
    else:
        return general_mm(a, b, out, M, N, K)
