import logging
import os

import torch
import triton
import triton.language as tl
import yaml

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

from .utils import (
    create_tma_device_descriptor,
    get_cached_tma_device_descriptor,
    should_enable_sqmma,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

SQMMA_SWITCH = True

EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "addmm_mthreads_expand.yaml")
)


def get_expand_config(op):
    default_strategies = {
        "addmm": ["default", "default", "default"],
        "addmm_sqmma": ["default", "default", "default"],
    }
    op_key_orders = {
        "addmm": ["M", "N", "K"],
        "addmm_sqmma": ["M", "N", "K"],
    }
    op_meta_map = {
        "addmm": {
            "BM": "BLOCK_SIZE_M",
            "BN": "BLOCK_SIZE_N",
            "BK": "BLOCK_SIZE_K",
        },
        "addmm_sqmma": {
            "BM": "BLOCK_SIZE_M",
            "BN": "BLOCK_SIZE_N",
            "BK": "BLOCK_SIZE_K",
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


def addmm_get_configs():
    if os.environ.get("USE_FLAGTUNE") == "1":
        expand_config = get_expand_config("addmm")
        if expand_config != -1:
            ranges = expand_config["ranges"]
            return [
                triton.Config(
                    {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK},
                    num_stages=s,
                    num_warps=w,
                )
                for BM in ranges["BM"]
                for BN in ranges["BN"]
                for BK in ranges["BK"]
                for s in ranges["s"]
                for w in ranges["w"]
            ]
    return runtime.get_tuned_config("addmm")


@libentry()
@libtuner(
    configs=addmm_get_configs(),
    key=["M", "N", "K"],
    strategy=get_expand_config("addmm")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1" and get_expand_config("addmm") != -1
    else ["default", "default", "default"],
    warmup=5,
    rep=5,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def addmm_fma(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_MTHREADS ADDMM(FMA)")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape).contiguous()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with torch_device_fn.device(mat1.device):
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias.stride(0),
            bias.stride(1),
            out.stride(0),
            out.stride(1),
        )
    return out


def addmm_sqmma_descriptor_pre_hook(nargs):
    a = nargs["A"]
    b = nargs["B"]
    bias = nargs["Bias"]
    c = nargs["C"]
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    device = c.device

    nargs["a_desc_ptr"].copy_(
        get_cached_tma_device_descriptor(a, block_m, block_k, device)
    )
    nargs["b_desc_ptr"].copy_(
        get_cached_tma_device_descriptor(b, block_k, block_n, device)
    )
    nargs["bias_desc_ptr"].copy_(
        get_cached_tma_device_descriptor(bias, block_m, block_n, device)
    )
    nargs["c_desc_ptr"].copy_(
        create_tma_device_descriptor(c, block_m, block_n, device)
    )


def addmm_sqmma_get_configs(pre_hook=addmm_sqmma_descriptor_pre_hook):
    if os.environ.get("USE_FLAGTUNE") == "1":
        expand_config = get_expand_config("addmm_sqmma")
        if expand_config != -1:
            ranges = expand_config["ranges"]
            return [
                triton.Config(
                    {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK},
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
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_stages=1,
            num_warps=4,
            pre_hook=pre_hook,
        )
    ]


@libentry()
@libtuner(
    configs=addmm_sqmma_get_configs(),
    key=["M", "N", "K"],
    strategy=get_expand_config("addmm_sqmma")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1" and get_expand_config("addmm_sqmma") != -1
    else ["default", "default", "default"],
    warmup=5,
    rep=5,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_sqmma_kernel(
    A,
    B,
    Bias,
    C,
    a_desc_ptr,
    b_desc_ptr,
    bias_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ab_type: tl.constexpr,
    c_type: tl.constexpr,
    is_transpose_a: tl.constexpr = False,
    is_transpose_b: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    offs_k = 0
    input_type = ab_type
    output_type = c_type
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl._experimental_descriptor_load(
            a_desc_ptr,
            [offs_am, offs_k],
            [BLOCK_SIZE_M, BLOCK_SIZE_K],
            input_type,
            is_transpose_a,
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr,
            [offs_k, offs_bn],
            [BLOCK_SIZE_K, BLOCK_SIZE_N],
            input_type,
            is_transpose_b,
        )
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_SIZE_K
    bias = tl._experimental_descriptor_load(
        bias_desc_ptr, [offs_am, offs_bn], [BLOCK_SIZE_M, BLOCK_SIZE_N], input_type
    )
    result = (alpha * accumulator.to(output_type) + beta * bias.to(output_type)).to(
        output_type
    )
    tl._experimental_descriptor_store(c_desc_ptr, result, [offs_am, offs_bn])


def get_triton_type(elem_type):
    type_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float8_e4m3fn: tl.float8e4nv,
    }
    return type_map.get(elem_type, None)


def addmm_sqmma(mat1, mat2, bias, elem_type, alpha, beta, M, N, K):
    logger.debug("GEMS_MTHREADS ADDMM(SQMMA)")
    device = mat1.device
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    # handle non-contiguous inputs if necessary
    is_transpose_a = False
    is_transpose_b = False
    if not mat1.is_contiguous():
        if mat1.stride(0) == 1 and mat1.stride(1) == mat1.shape[0]:
            is_transpose_a = True
        else:
            mat1 = mat1.contiguous()
    if not mat2.is_contiguous():
        if mat2.stride(0) == 1 and mat2.stride(1) == mat2.shape[0]:
            is_transpose_b = True
        else:
            mat2 = mat2.contiguous()
    ab_type = elem_type
    a_type = mat1.dtype
    b_type = mat2.dtype
    assert a_type == b_type, "Mat A and Mat B should have the same dtype"
    c_type = a_type
    C = torch.empty((M, N), dtype=c_type, device=device)
    bias = bias.broadcast_to(C.shape).contiguous()
    desc_a = torch.empty((64,), dtype=torch.int8, device=device)
    desc_b = torch.empty((64,), dtype=torch.int8, device=device)
    desc_bias = torch.empty((64,), dtype=torch.int8, device=device)
    desc_c = torch.empty((64,), dtype=torch.int8, device=device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        1,
        1,
    )
    addmm_sqmma_kernel[grid](
        mat1,
        mat2,
        bias,
        C,
        desc_a,
        desc_b,
        desc_bias,
        desc_c,
        M,
        N,
        K,
        alpha,
        beta,
        ab_type=get_triton_type(ab_type),
        c_type=get_triton_type(c_type),
        is_transpose_a=is_transpose_a,
        is_transpose_b=is_transpose_b,
    )
    return C


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    a_dtype = mat1.dtype
    b_dtype = mat2.dtype
    M, K = mat1.shape
    _, N = mat2.shape
    use_sqmma = should_enable_sqmma(a_dtype, b_dtype, M, N, K) and SQMMA_SWITCH

    if use_sqmma:
        return addmm_sqmma(
            mat1,
            mat2,
            bias,
            a_dtype,
            alpha,
            beta,
            M,
            N,
            K,
        )
    else:
        enable_sqmma = os.environ.pop("MUSA_ENABLE_SQMMA", None)
        result = addmm_fma(bias, mat1, mat2, alpha=alpha, beta=beta)
        if enable_sqmma:
            os.environ["MUSA_ENABLE_SQMMA"] = enable_sqmma
        return result
