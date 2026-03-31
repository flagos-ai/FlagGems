import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl
import yaml

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(
    "flag_gems.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_matmul"
)
CACHE_USAGE_THRESHOLD = 0.8
EXPAND_CONFIG_FILENAME = "w8a8_block_fp8_matmul_hopper_expand.yaml"


@functools.lru_cache
def get_w8a8_block_fp8_hopper_configs(
    N: int, K: int, block_n: int, block_k: int
) -> Optional[Dict[int, Any]]:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    name_parts = device_name.split("_")
    if any(part.startswith("H200") for part in name_parts):
        device_name = "NVIDIA_H200"
    elif any(part.startswith("H20") for part in name_parts):
        device_name = "NVIDIA_H20"
    file_name = f"fa8_w8a8-{block_n}-{block_k}.yaml"

    config_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
        "utils",
        "configs",
    )
    cfg_file = os.path.join(config_dir, file_name)

    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            logger.info(
                "Using config from %s for W8A8 block FP8 kernel.",
                cfg_file,
            )
            dev_data = yaml.safe_load(f).get(device_name, {})
            NK_data = dev_data.get(f"{N},{K}", {})

            result = {}
            for k, p in NK_data.items():
                # unpack the list into dictionary
                result[int(k)] = {
                    "BLOCK_SIZE_M": p[0],
                    "BLOCK_SIZE_N": p[1],
                    "BLOCK_SIZE_K": p[2],
                    "GROUP_SIZE_M": p[3],
                    "num_warps": p[4],
                    "num_stages": p[5],
                }

            if not result:
                return None
            return result

    logger.warning(
        "Using default W8A8 Block FP8 kernel config. Performance might "
        "be sub-optimal! Config file not found at %s",
        cfg_file,
    )
    return None


def _build_fixed_matmul_config(
    config: Dict[str, int], pre_hook=None
) -> triton.Config:
    return triton.Config(
        {
            "BLOCK_M": config["BLOCK_SIZE_M"],
            "BLOCK_N": config["BLOCK_SIZE_N"],
            "BLOCK_K": config["BLOCK_SIZE_K"],
            "GROUP_M": config["GROUP_SIZE_M"],
        },
        num_stages=config["num_stages"],
        num_warps=config["num_warps"],
        pre_hook=pre_hook,
    )


@contextmanager
def _use_fixed_matmul_configs(config: Dict[str, int]):
    general_tuner = w8a8_block_fp8_matmul_kernel_general.fn
    host_tma_tuner = w8a8_block_fp8_matmul_kernel_host_tma.fn
    general_configs = general_tuner.configs
    host_tma_configs = host_tma_tuner.configs
    general_tuner.configs = [_build_fixed_matmul_config(config)]
    host_tma_tuner.configs = [
        _build_fixed_matmul_config(
            config,
            pre_hook=matmul_tma_set_block_size_hook,
        )
    ]
    try:
        yield
    finally:
        general_tuner.configs = general_configs
        host_tma_tuner.configs = host_tma_configs


def is_tma_compatible(a, b, n, k):
    """
    Check if tensors are compatible with TMA (Tensor Memory Accelerator).

    TMA requires 128-bit (16-byte) alignment for memory access.
    For FP8 inputs (1 byte/element), both N and K must be multiples of 16
    to satisfy the 16-byte alignment requirement.

    Args:
        a, b: Input tensors
        n, k: Matrix dimensions

    Returns:
        bool: True if compatible with TMA's 128-bit alignment requirement
    """
    return (
        a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and b.dtype == a.dtype
        and n % 16 == 0
        and k % 16 == 0
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
        # B is stored as [N, K] in row-major order, and the kernel loads an
        # [BLOCK_N, BLOCK_K] tile before transposing it to [BLOCK_K, BLOCK_N].
        nargs["b_desc"].block_shape = [BLOCK_N, BLOCK_K]
    else:
        # For the column-major case we build the descriptor on B.T with shape
        # [K, N], so the loaded tile already has layout [BLOCK_K, BLOCK_N].
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]

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
    config_path = os.path.join(os.path.dirname(__file__), "..", EXPAND_CONFIG_FILENAME)
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
                "Using expand configurations from %s for matmul kernel autotuning",
                EXPAND_CONFIG_FILENAME,
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
    configs=matmul_get_configs(pre_hook=None)
    if os.environ.get("USE_FLAGTUNE") == "1" and get_expand_config("matmul") != -1
    else runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    strategy=get_expand_config("matmul")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1" and get_expand_config("matmul") != -1
    else ["default", "default", "default", "default", "default", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def w8a8_block_fp8_matmul_kernel_general(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_n,
    stride_Bs_k,
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
    pid_n = (pid % width) // group_size

    if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0:
        # offset
        offset_am = pid_m * BLOCK_M
        offset_bn = pid_n * BLOCK_N
        offs_am = offset_am + tl.arange(0, BLOCK_M)
        offs_bn = offset_bn + tl.arange(0, BLOCK_N)
        offset_k = 0

        a_desc = tl.make_tensor_descriptor(
            base=A,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_M, BLOCK_K],
        )

        # B is stored as [N, K] in row-major order for w8a8 block FP8 matmul.
        b_desc = tl.make_tensor_descriptor(
            base=B,
            shape=[N, K],
            strides=[K, 1],
            block_shape=[BLOCK_N, BLOCK_K],
        )
        c_desc = tl.make_tensor_descriptor(
            base=C,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_M, BLOCK_N],
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([offset_am.to(tl.int32), offset_k.to(tl.int32)])
            b_t = b_desc.load([offset_bn.to(tl.int32), offset_k.to(tl.int32)])
            b = tl.trans(b_t)
            offs_ks = (offset_k // group_k).to(tl.int32)
            a_s = tl.load(As + offs_am * stride_As_m + offs_ks * stride_As_k)
            b_s = tl.load(
                Bs + (offs_bn // group_n) * stride_Bs_n + offs_ks * stride_Bs_k
            )
            acc += tl.dot(a, b, out_dtype=tl.float32) * a_s[:, None] * b_s[None, :]
            offset_k += BLOCK_K

        c_desc.store(
            [offset_am.to(tl.int32), offset_bn.to(tl.int32)], acc.to(c_desc.dtype)
        )
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
        prev_multiple = prev_multiple_of(K, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start_k in range(0, prev_multiple, BLOCK_K):
            rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
            a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
            b_t = tl.load(B + (rbn[:, None] * stride_bn + rk[None, :] * stride_bk))
            b = tl.trans(b_t)
            offs_ks = start_k // group_k
            a_s = tl.load(As + ram * stride_As_m + offs_ks * stride_As_k)
            b_s = tl.load(Bs + (rbn // group_n) * stride_Bs_n + offs_ks * stride_Bs_k)
            acc += tl.dot(a, b, out_dtype=tl.float32) * a_s[:, None] * b_s[None, :]

        # loop peeling
        rk = (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = rk < K
        a = tl.load(
            A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b_t = tl.load(
            B + (rbn[:, None] * stride_bn + rk[None, :] * stride_bk),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.trans(b_t)
        offs_ks = prev_multiple // group_k
        a_s = tl.load(As + ram * stride_As_m + offs_ks * stride_As_k)
        b_s = tl.load(Bs + (rbn // group_n) * stride_Bs_n + offs_ks * stride_Bs_k)
        acc += tl.dot(a, b, out_dtype=tl.float32) * a_s[:, None] * b_s[None, :]

        # rematerialize rm and rn to save registers
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        c_ptrs = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # handles write-back with reduction-splitting
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=matmul_get_configs(),
    key=["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    strategy=get_expand_config("matmul")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1" and get_expand_config("matmul") != -1
    else ["align32", "align32", "align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
)
@triton.jit
def w8a8_block_fp8_matmul_kernel_host_tma(
    a_desc,
    b_desc,
    c_desc,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_n,
    stride_Bs_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
    dtype: tl.constexpr,
    enable_warp_specialization=True,
):
    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offset_am = (pid_m * BLOCK_M).to(tl.int32)
    offset_bn = (pid_n * BLOCK_N).to(tl.int32)
    offs_am = offset_am + tl.arange(0, BLOCK_M)
    offs_bn = offset_bn + tl.arange(0, BLOCK_N)
    iters = tl.cdiv(K, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(iters):
        offset_ak = (k * BLOCK_K).to(tl.int32)

        if A_ROW_MAJOR:
            a = a_desc.load([offset_am, offset_ak])
        else:
            a_t = a_desc.load([offset_ak, offset_am])
            a = tl.trans(a_t)

        if B_ROW_MAJOR:
            b_t = b_desc.load([offset_bn, offset_ak])
            b = tl.trans(b_t)
        else:
            b = b_desc.load([offset_ak, offset_bn])

        offs_ks = (offset_ak // group_k).to(tl.int32)
        a_s = tl.load(
            As + offs_am * stride_As_m + offs_ks * stride_As_k,
            mask=offs_am < M,
            other=0.0,
        )
        b_s = tl.load(
            Bs + (offs_bn // group_n) * stride_Bs_n + offs_ks * stride_Bs_k,
            mask=offs_bn < N,
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32) * a_s[:, None] * b_s[None, :]

    c_desc.store([offset_am, offset_bn], acc.to(c_desc.dtype))


def gemv_get_configs():
    if os.environ.get("USE_FLAGTUNE") == "1":
        expand_config = get_expand_config("gemv")
        if expand_config != -1:
            logger.debug(
                "Using expand configurations from %s for gemv kernel autotuning",
                EXPAND_CONFIG_FILENAME,
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
    configs=gemv_get_configs(),
    key=["M", "K", "stride_am", "stride_bk"],
    strategy=get_expand_config("gemv")["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1" and get_expand_config("gemv") != -1
    else ["align32", "align32", "align32", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def w8a8_block_fp8_matmul_gemv_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    K,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
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
        offs_ks = k_start // group_k
        a_s = tl.load(
            As + row_offset * stride_As_m + offs_ks * stride_As_k,
            mask=row_mask,
            other=0.0,
        )
        b_s = tl.load(Bs + offs_ks * stride_Bs_k)

        # Dequantize each K block on the fly and accumulate in fp32.
        partial = tl.sum(a.to(tl.float32) * b[None, :].to(tl.float32), axis=1)
        acc += partial * a_s * b_s

    # Store result
    tl.store(C + row_offset, acc.to(C.dtype.element_ty), mask=row_mask)


def general_w8a8_block_fp8_matmul(a, b, c, a_s, b_s, M, N, K, group_n, group_k):
    logger.debug(
        "GEMS w8a8_block_fp8_matmul-hopper, [scenario]: general, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )
    dtype_str = str(a.dtype).split(".")[-1]
    fixed_config = None
    if os.environ.get("USE_FLAGTUNE") != "1":
        configs = get_w8a8_block_fp8_hopper_configs(N, K, group_n, group_k)
        if configs:
            fixed_config = configs[min(configs.keys(), key=lambda x: abs(x - M))]

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
            a_desc = TensorDescriptor(a.T, a.T.shape, a.T.stride(), dummy_block)

        if b_row_major:
            b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
        else:
            b_desc = TensorDescriptor(b.T, b.T.shape, b.T.stride(), dummy_block)

        c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
        kernel_kwargs = {
            "GROUP_M": 8,
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            "dtype": dtype_str,
        }
        if fixed_config is not None:
            kernel_kwargs.pop("GROUP_M")

        launch = lambda: w8a8_block_fp8_matmul_kernel_host_tma[grid](
            a_desc,
            b_desc,
            c_desc,
            a_s,
            b_s,
            M,
            N,
            K,
            group_n,
            group_k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            a_s.stride(0),
            a_s.stride(1),
            b_s.stride(0),
            b_s.stride(1),
            **kernel_kwargs,
        )

        if fixed_config is not None:
            with _use_fixed_matmul_configs(fixed_config):
                with torch_device_fn.device(a.device):
                    launch()
        else:
            with torch_device_fn.device(a.device):
                launch()
    else:

        def alloc_fn(size: int, align: int, stream: Optional[int]):
            return torch.empty(size, dtype=torch.int8, device=a.device)

        triton.set_allocator(alloc_fn)
        kernel_kwargs = {
            "dtype": dtype_str,
            "GROUP_M": 8,
        }
        if fixed_config is not None:
            kernel_kwargs.pop("GROUP_M")

        launch = lambda: w8a8_block_fp8_matmul_kernel_general[grid](
            a,
            b,
            c,
            a_s,
            b_s,
            M,
            N,
            K,
            group_n,
            group_k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            a_s.stride(0),
            a_s.stride(1),
            b_s.stride(0),
            b_s.stride(1),
            **kernel_kwargs,
        )
        if fixed_config is not None:
            with _use_fixed_matmul_configs(fixed_config):
                with torch_device_fn.device(a.device):
                    launch()
        else:
            with torch_device_fn.device(a.device):
                launch()
    return c


def gemv_w8a8_block_fp8_matmul(a, b, c, a_s, b_s, M, K, group_k):
    """Optimized matrix-vector multiplication for N=1 case"""
    logger.debug(
        "GEMS w8a8_block_fp8_matmul-hopper, [scenario]: gemv (N=1), [shape info]: [%s, %s, 1](M, K, N)",
        M,
        K,
    )
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)

    with torch_device_fn.device(a.device):
        w8a8_block_fp8_matmul_gemv_kernel[grid](
            a,
            b,
            c,
            a_s,
            b_s,
            M,
            K,
            group_k,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            a_s.stride(0),
            a_s.stride(1),
            b_s.stride(1),
        )
    return c


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    device = A.device
    assert len(block_size) == 2
    block_n, block_k = block_size

    # handle non-contiguous inputs if necessary
    if A.ndim >= 2 and A.stride(-2) > 1 and A.stride(-1) > 1:
        A = A.contiguous()
    if B.ndim == 2 and B.stride(0) > 1 and B.stride(1) > 1:
        B = B.contiguous()
    if As.ndim >= 2 and As.stride(-2) > 1 and As.stride(-1) > 1:
        As = As.contiguous()
    if Bs.ndim == 2 and Bs.stride(0) > 1 and Bs.stride(1) > 1:
        Bs = Bs.contiguous()

    # checks constraints
    assert A.shape[-1] == B.shape[-1], "incompatible dimensions"
    assert A.shape[:-1] == As.shape[:-1], "A and As dimensions mismatch"
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1], "invalid As shape"
    assert B.ndim == 2 and Bs.ndim == 2, "B and Bs must be 2D"

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0], "invalid Bs N dimension"
    assert triton.cdiv(K, block_k) == Bs.shape[1], "invalid Bs K dimension"

    # allocates output
    output_shape = A.shape[:-1] + (N,)
    c = torch.empty(output_shape, device=device, dtype=output_dtype)

    a_2d = A.reshape(M, K)
    as_2d = As.reshape(M, As.shape[-1])
    c_2d = c.reshape(M, N)

    # Optimize for N=1 case (matrix-vector multiplication)
    if N == 1:
        return gemv_w8a8_block_fp8_matmul(
            a_2d,
            B,
            c if c.ndim == 1 else c.squeeze(-1),
            as_2d,
            Bs,
            M,
            K,
            block_k,
        ).reshape(c.shape)

    return general_w8a8_block_fp8_matmul(
        a_2d,
        B,
        c_2d,
        as_2d,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
    ).reshape(c.shape)
