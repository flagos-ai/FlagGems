import functools
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl
import yaml

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(
    "flag_gems.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_matmul"
)
CACHE_USAGE_THRESHOLD = 0.8
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "w8a8_block_fp8_matmul_hopper_expand.yaml",
    )
)


@functools.lru_cache
def get_w8a8_block_fp8_hopper_configs(N: int, K: int) -> Optional[Dict[int, Any]]:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    file_name = "w8a8_block_fp8_matmul_hopper.yaml"

    cfg_file = os.path.join(os.path.dirname(__file__), "..", file_name)

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


def _get_placeholder_tuner_configs(pre_hook=None):
    # Placeholder config for libtuner initialization before runtime shapes are known.
    return [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
            },
            num_stages=3,
            num_warps=4,
            pre_hook=pre_hook,
        )
    ]


def _get_fixed_matmul_meta(M: int, N: int, K: int, block_n: int, block_k: int):
    configs = get_w8a8_block_fp8_hopper_configs(N, K)
    if not configs:
        return {
            "BLOCK_M": 64,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    return {
        "BLOCK_M": config["BLOCK_SIZE_M"],
        "BLOCK_N": config["BLOCK_SIZE_N"],
        "BLOCK_K": config["BLOCK_SIZE_K"],
        "GROUP_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


@libentry()
@libtuner(
    configs=runtime.ops_get_configs(
        "w8a8_block_fp8_general",
        pre_hook=None,
        yaml_path=EXPAND_CONFIG_FILENAME,
    )
    if os.environ.get("USE_FLAGTUNE") == "1"
    else _get_placeholder_tuner_configs(pre_hook=None),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=runtime.get_expand_config(
        "w8a8_block_fp8_general", yaml_path=EXPAND_CONFIG_FILENAME
    )["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=5,
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
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        k_start = k * BLOCK_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)
        acc += tl.dot(a, b, out_dtype=tl.float32) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = acc.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = acc.to(tl.float16)
    else:
        c = acc.to(tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@libentry()
@libtuner(
    configs=runtime.ops_get_configs(
        "w8a8_block_fp8_general_splitk",
        yaml_path=EXPAND_CONFIG_FILENAME,
    )
    if os.environ.get("USE_FLAGTUNE") == "1"
    else _get_placeholder_tuner_configs(pre_hook=None),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=runtime.get_expand_config(
        "w8a8_block_fp8_general_splitk", yaml_path=EXPAND_CONFIG_FILENAME
    )["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=5,
)
@triton.jit
def w8a8_block_fp8_matmul_kernel_splitk(
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
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    # grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offset_am = pid_m * BLOCK_M
    offset_bn = pid_n * BLOCK_N
    offs_am = offset_am + tl.arange(0, BLOCK_M)
    offs_bn = offset_bn + tl.arange(0, BLOCK_N)

    total_k_iters = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(total_k_iters, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min((pid_k + 1) * k_per_split, total_k_iters)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end):
        offset_k = k * BLOCK_K
        offs_k = offset_k + tl.arange(0, BLOCK_K)

        a = tl.load(
            A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N),
            other=0.0,
        )

        offs_ks = offset_k // group_k
        a_s = tl.load(
            As + offs_am * stride_As_m + offs_ks * stride_As_k,
            mask=offs_am < M,
            other=0.0,
        )
        b_s = tl.load(
            Bs + offs_ks * stride_Bs_k + (offs_bn // group_n) * stride_Bs_n,
            mask=offs_bn < N,
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32) * a_s[:, None] * b_s[None, :]

    offs_cm = offset_am + tl.arange(0, BLOCK_M)
    offs_cn = offset_bn + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    if C.dtype.element_ty == tl.bfloat16:
        tl.atomic_add(c_ptrs, acc.to(tl.bfloat16), mask=mask)
    elif C.dtype.element_ty == tl.float16:
        tl.atomic_add(c_ptrs, acc.to(tl.float16), mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc.to(tl.float32), mask=mask)


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

    use_flagtune = os.environ.get("USE_FLAGTUNE") == "1"

    # Split-K path for small-N, large-K shapes
    if M < 2048 and N < 2112 and K >= 4096:
        if use_flagtune:
            splitk_grid = lambda META: (
                triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
                META["SPLIT_K"],
            )
            c.zero_()
            with torch_device_fn.device(a.device):
                w8a8_block_fp8_matmul_kernel_splitk[splitk_grid](
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
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    a_s.stride(0),
                    a_s.stride(1),
                    b_s.stride(1),
                    b_s.stride(0),
                )
        else:
            SPLITK_BLOCK_K = 128
            SPLITK_BLOCK_M = 16 if M <= 16 else 64
            SPLITK_BLOCK_N = 64 if N > 256 else 32

            grid_m = triton.cdiv(M, SPLITK_BLOCK_M)
            grid_n = triton.cdiv(N, SPLITK_BLOCK_N)
            grid_mn = grid_m * grid_n
            total_k_iters = triton.cdiv(K, SPLITK_BLOCK_K)

            SM_COUNT = torch.cuda.get_device_properties(a.device).multi_processor_count
            split_k = min(total_k_iters, max(4, 2 * SM_COUNT // max(grid_mn, 1)))

            c.zero_()
            splitk_grid = (grid_mn, split_k)

            with torch_device_fn.device(a.device):
                w8a8_block_fp8_matmul_kernel_splitk.fn.fn[splitk_grid](
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
                    b.stride(1),
                    b.stride(0),
                    c.stride(0),
                    c.stride(1),
                    a_s.stride(0),
                    a_s.stride(1),
                    b_s.stride(1),
                    b_s.stride(0),
                    BLOCK_M=SPLITK_BLOCK_M,
                    BLOCK_N=SPLITK_BLOCK_N,
                    BLOCK_K=SPLITK_BLOCK_K,
                    SPLIT_K=split_k,
                )
        return c

    else:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        fixed_meta = (
            None
            if use_flagtune
            else _get_fixed_matmul_meta(M, N, K, block_n=group_n, block_k=group_k)
        )

        def alloc_fn(size: int, align: int, stream: Optional[int]):
            return torch.empty(size, dtype=torch.int8, device=a.device)

        triton.set_allocator(alloc_fn)
        if use_flagtune:
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
                b.stride(1),
                b.stride(0),
                c.stride(0),
                c.stride(1),
                a_s.stride(0),
                a_s.stride(1),
                b_s.stride(1),
                b_s.stride(0),
            )
        else:
            launch = lambda: w8a8_block_fp8_matmul_kernel_general.fn.fn[grid](
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
                b.stride(1),
                b.stride(0),
                c.stride(0),
                c.stride(1),
                a_s.stride(0),
                a_s.stride(1),
                b_s.stride(1),
                b_s.stride(0),
                **fixed_meta,
            )

        with torch_device_fn.device(a.device):
            launch()
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
