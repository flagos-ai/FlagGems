import functools
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import triton
import triton.language as tl
import triton_tvm_ffi
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

TMA_ON = False


@functools.lru_cache
def get_w8a8_block_fp8_hopper_configs(N: int, K: int) -> Optional[Dict[int, Any]]:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    name_parts = device_name.split("_")
    if any(part.startswith("H20") for part in name_parts):
        device_name = "NVIDIA_H20"
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
        and TMA_ON
        and n % 16 == 0
        and k % 16 == 0
    )


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


@triton_tvm_ffi.jit
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
@triton.jit(
    do_not_specialize=[
        "M",
        "N",
        "K",
        "group_n",
        "group_k",
        "stride_am",
        "stride_ak",
        "stride_bk",
        "stride_bn",
        "stride_cm",
        "stride_cn",
        "stride_As_m",
        "stride_As_k",
        "stride_Bs_k",
        "stride_Bs_n",
    ]
)
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
        "w8a8_block_fp8_general_tma",
        pre_hook=matmul_tma_set_block_size_hook,
        yaml_path=EXPAND_CONFIG_FILENAME,
    )
    if os.environ.get("USE_FLAGTUNE") == "1"
    else _get_placeholder_tuner_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    strategy=runtime.get_expand_config(
        "w8a8_block_fp8_general_tma", yaml_path=EXPAND_CONFIG_FILENAME
    )["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
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


@triton_tvm_ffi.torch_wrap(
    [w8a8_block_fp8_matmul_kernel_general],
    Path(__file__).with_suffix(".cc"),
    extra_include_paths=[
        str(Path(triton_tvm_ffi.__file__).resolve().parents[2] / "include")
    ],
)
def w8a8_block_fp8_matmul_general_tvm_ffi(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    M: int,
    N: int,
    K: int,
    group_n: int,
    group_k: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    stride_As_m: int,
    stride_As_k: int,
    stride_Bs_k: int,
    stride_Bs_n: int,
) -> torch.Tensor:
    ...


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
    if hasattr(
        triton.tools.tensor_descriptor, "TensorDescriptor"
    ) and is_tma_compatible(a, b, N, K):
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        use_flagtune = os.environ.get("USE_FLAGTUNE") == "1"
        fixed_meta = (
            None
            if use_flagtune
            else _get_fixed_matmul_meta(M, N, K, block_n=group_n, block_k=group_k)
        )
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
        if use_flagtune:
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
                A_ROW_MAJOR=a_row_major,
                B_ROW_MAJOR=b_row_major,
                dtype=str(a.dtype).split(".")[-1],
            )
        else:
            # The fixed-config path bypasses libtuner, so we must apply the
            # descriptor block-shape update that would normally run via the
            # TMA pre_hook before launching the underlying JIT kernel.
            matmul_tma_set_block_size_hook(
                {
                    "BLOCK_M": fixed_meta["BLOCK_M"],
                    "BLOCK_N": fixed_meta["BLOCK_N"],
                    "BLOCK_K": fixed_meta["BLOCK_K"],
                    "a_desc": a_desc,
                    "b_desc": b_desc,
                    "c_desc": c_desc,
                    "A_ROW_MAJOR": a_row_major,
                    "B_ROW_MAJOR": b_row_major,
                }
            )
            launch = lambda: w8a8_block_fp8_matmul_kernel_host_tma.fn.fn[grid](
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
                A_ROW_MAJOR=a_row_major,
                B_ROW_MAJOR=b_row_major,
                dtype=str(a.dtype).split(".")[-1],
                **fixed_meta,
            )

        with torch_device_fn.device(a.device):
            launch()
    else:
        with torch_device_fn.device(a.device):
            w8a8_block_fp8_matmul_general_tvm_ffi(
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
