import logging
import os
from typing import List

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(
    "flag_gems.runtime.backend._mthreads.ops.w8a8_block_fp8_matmul"
)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "w8a8_block_fp8_matmul_mthreads_expand.yaml",
    )
)

SQMMA_ON = False


def is_supported_sqmma_layout(tensor):
    return tensor.is_contiguous() or (
        tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
    )


def is_sqmma_compatible(a, b, output_dtype, n, k):
    return (
        a.dim() == 2
        and SQMMA_ON
        and b.dim() == 2
        and a.dtype == b.dtype == torch.float8_e4m3fn
        and output_dtype in (torch.float16, torch.bfloat16)
        and is_supported_sqmma_layout(a)
        and is_supported_sqmma_layout(b)
        and n % 16 == 0
        and k % 16 == 0
    )


def get_triton_type(elem_type):
    type_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float8_e4m3fn: tl.float8e4nv,
    }
    return type_map.get(elem_type, None)


def matmul_get_configs():
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
        )
    ]


@libentry()
@libtuner(
    configs=runtime.ops_get_configs(
        "w8a8_block_fp8_general", pre_hook=None, yaml_path=EXPAND_CONFIG_FILENAME
    )
    if os.environ.get("USE_FLAGTUNE") == "1"
    else matmul_get_configs(),
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
def w8a8_block_fp8_matmul_kernel(
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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        k_start = k * BLOCK_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def w8a8_block_fp8_matmul_sqmma_kernel(
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
    stride_As_m,
    stride_As_k,
    stride_Bs_n,
    stride_Bs_k,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = (pid_m * BLOCK_M).to(tl.int32)
    offs_bn = (pid_n * BLOCK_N).to(tl.int32)
    offs_k = tl.zeros((), dtype=tl.int32)

    row_offset = offs_am + tl.arange(0, BLOCK_M)
    col_offset = offs_bn + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(b_desc, [offs_k, offs_bn])

        scale_k = offs_k // group_k
        a_s = tl.load(
            As + row_offset * stride_As_m + scale_k * stride_As_k,
            mask=row_offset < M,
            other=0.0,
        )
        b_s = tl.load(
            Bs + (col_offset // group_n) * stride_Bs_n + scale_k * stride_Bs_k,
            mask=col_offset < N,
            other=0.0,
        )
        acc += (
            tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
            * a_s[:, None]
            * b_s[None, :]
        )
        offs_k += BLOCK_K

    tl.store_tensor_descriptor(c_desc, [offs_am, offs_bn], acc.to(c_desc.dtype))


def general_w8a8_block_fp8_matmul(
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
):
    logger.debug(
        "GEMS_MTHREADS W8A8_BLOCK_FP8_MATMUL(general), [shape info]: [-, %s, %s, %s](batch, M, N, K)",
        M,
        N,
        K,
    )
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(a.device):
        w8a8_block_fp8_matmul_kernel[grid](
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


def sqmma_w8a8_block_fp8_matmul(
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
):
    logger.debug(
        "GEMS_MTHREADS W8A8_BLOCK_FP8_MATMUL(sqmma), [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s",
        M,
        N,
        K,
        a.stride(0) == 1,
        b.stride(0) == 1,
    )
    device = a.device
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 128
    GROUP_M = 8

    desc_a = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
    desc_b = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
    desc_c = TensorDescriptor.from_tensor(c, [BLOCK_M, BLOCK_N])

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        1,
        1,
    )

    with torch_device_fn.device(device):
        w8a8_block_fp8_matmul_sqmma_kernel[grid](
            desc_a,
            desc_b,
            desc_c,
            a_s,
            b_s,
            M,
            N,
            K,
            group_n,
            group_k,
            a_s.stride(0),
            a_s.stride(1),
            b_s.stride(0),
            b_s.stride(1),
            GROUP_M,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
    return c


def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    device = A.device
    assert len(block_size) == 2
    block_n, block_k = block_size

    if A.ndim >= 2 and A.stride(-2) > 1 and A.stride(-1) > 1:
        A = A.contiguous()
    if B.ndim == 2 and B.stride(0) > 1 and B.stride(1) > 1:
        B = B.contiguous()
    if As.ndim >= 2 and As.stride(-2) > 1 and As.stride(-1) > 1:
        As = As.contiguous()
    if Bs.ndim == 2 and Bs.stride(0) > 1 and Bs.stride(1) > 1:
        Bs = Bs.contiguous()

    assert A.shape[-1] == B.shape[-1], "incompatible dimensions"
    assert A.shape[:-1] == As.shape[:-1], "A and As dimensions mismatch"
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1], "invalid As shape"
    assert B.ndim == 2 and Bs.ndim == 2, "B and Bs must be 2D"

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0], "invalid Bs N dimension"
    assert triton.cdiv(K, block_k) == Bs.shape[1], "invalid Bs K dimension"

    output_shape = A.shape[:-1] + (N,)
    c = torch.empty(output_shape, device=device, dtype=output_dtype)

    a_2d = A.reshape(M, K)
    as_2d = As.reshape(M, As.shape[-1])
    c_2d = c.reshape(M, N)
    if is_sqmma_compatible(a_2d, B, output_dtype, N, K):
        return sqmma_w8a8_block_fp8_matmul(
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
