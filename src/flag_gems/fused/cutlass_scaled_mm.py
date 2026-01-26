from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

SCALE_BLOCK_K, SCALE_BLOCK_N = 128, 128


def get_sm_version_num():
    if hasattr(torch_device_fn, "get_device_capability"):
        major, minor = torch_device_fn.get_device_capability()
        return major * 10 + minor


SM_VERSION_NUM = get_sm_version_num()


def get_block_wise_smm_configs():
    tile_configs = [
        # (TILE_M, TILE_N, num_stages, num_warps)
        (32, 64, 5, 2),
        (64, 32, 5, 2),
        (64, 128, 4, 4),
        (64, 256, 4, 4),
        (128, 32, 4, 4),
        (128, 64, 4, 4),
        (128, 128, 4, 4),
        (128, 256, 3, 8),
        (256, 64, 4, 4),
        (256, 128, 3, 8),
    ]

    return [
        triton.Config(
            {
                "TILE_M": TILE_M,
                "TILE_N": TILE_N,
                "TILE_K": SCALE_BLOCK_K,
                "SWIZZLE_GROUP_M": 8,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for TILE_M, TILE_N, stages, warps in tile_configs
    ]


@triton.jit
def grouped_launch(
    pid, M, N, TILE_M: tl.constexpr, TILE_N: tl.constexpr, SWIZZLE_GROUP_M: tl.constexpr
):
    grid_m = tl.cdiv(M, TILE_M)
    grid_n = tl.cdiv(N, TILE_N)

    width = SWIZZLE_GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * SWIZZLE_GROUP_M, SWIZZLE_GROUP_M)

    pid_m = group_id * SWIZZLE_GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


# block-wise dequantization kernel implemention
# this kernel supports many `SCALE_BLOCK_K, SCALE_BLOCK_N` cases
# as long as `TILE_K == SCALE_BLOCK_K` and `TILE_N % SCALE_BLOCK_N == 0`
@triton.autotune(
    configs=get_block_wise_smm_configs(),
    key=["_M_NPO2", "N", "K"],
)
@triton.jit
def _block_wise_smm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    _M_NPO2: tl.constexpr,
    SCALE_BLOCK_N,
    SCALE_BLOCK_K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_Ascale_m,
    stride_Ascale_k,
    stride_Bscale_k,
    stride_Bscale_n,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    SWIZZLE_GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = grouped_launch(pid, M, N, TILE_M, TILE_N, SWIZZLE_GROUP_M)

    offs_am = (pid_m * TILE_M + tl.arange(0, TILE_M)) % M
    offs_bn = (pid_n * TILE_N + tl.arange(0, TILE_N)) % N
    offs_k = tl.arange(0, TILE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    a_scale_ptrs = a_scale_ptr + offs_am * stride_Ascale_m
    offs_bsn = offs_bn // SCALE_BLOCK_N
    b_scale_ptrs = b_scale_ptr + offs_bsn * stride_Bscale_n

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        k_remaining = K - k * TILE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        offs_ks = k * TILE_K // SCALE_BLOCK_K
        a_scale = tl.load(a_scale_ptrs + offs_ks * stride_Ascale_k)
        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_Bscale_k)
        acc += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    acc = acc.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_cn = pid_n * TILE_N + tl.arange(0, TILE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def _block_wise_128_smm_launcher(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    global SCALE_BLOCK_K, SCALE_BLOCK_N
    SCALE_BLOCK_K, SCALE_BLOCK_N = 128, 128
    M, K = a.shape
    _, N = b.shape
    _M_NPO2 = triton.next_power_of_2(M)

    grid = lambda META: (
        triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
    )

    _block_wise_smm_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        M,
        N,
        K,
        _M_NPO2,
        SCALE_BLOCK_N,
        SCALE_BLOCK_K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
    )

    return c


# per-tensor and per-token dequantization kernel implemention
@triton.autotune(
    configs=[
        triton.Config({"TILE_M": 64, "TILE_N": 64, "TILE_K": 256}),
        triton.Config({"TILE_M": 64, "TILE_N": 128, "TILE_K": 128}),
        triton.Config({"TILE_M": 128, "TILE_N": 128, "TILE_K": 128}),
    ],
    key=["_M_NPO2", "N", "K"],
)
@triton.jit
def _pertensor_or_pertoken_smm_kernel(
    c_ptr,
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    _M_NPO2,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACC_DTYPE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    IS_PER_TOKEN_A: tl.constexpr,
    IS_PER_TOKEN_B: tl.constexpr,
):
    if IS_PER_TOKEN_A:
        TILE_SIZE_SCALE_A: tl.constexpr = TILE_M
    else:
        TILE_SIZE_SCALE_A: tl.constexpr = 1

    if IS_PER_TOKEN_B:
        TILE_SIZE_SCALE_B: tl.constexpr = TILE_N
    else:
        TILE_SIZE_SCALE_B: tl.constexpr = 1

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, TILE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    acc = tl.zeros((TILE_M, TILE_N), dtype=ACC_DTYPE)

    offsets_am = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, TILE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    offsets_scale_am = (
        tl.arange(0, TILE_SIZE_SCALE_A) + (TILE_SIZE_SCALE_A > 1) * pid_m * TILE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, TILE_SIZE_SCALE_B) + (TILE_SIZE_SCALE_B > 1) * pid_n * TILE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = a_scale_ptr + offsets_scale_am
    scale_b_ptrs = b_scale_ptr + offsets_scale_bn

    for k in range(0, tl.cdiv(K, TILE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        acc = tl.dot(a, b, acc, out_dtype=ACC_DTYPE)

        offsets_k += TILE_K
        a_ptrs += TILE_K * stride_ak
        b_ptrs += TILE_K * stride_bk

    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    a_scale = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    a_scale = a_scale.broadcast_to((TILE_M, 1))
    acc = a_scale * acc.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    b_scale = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    b_scale = b_scale.broadcast_to((TILE_N, 1))
    acc = b_scale.T * acc.to(tl.float32)

    c = acc.to(c_ptr.type.element_ty)

    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    offs_cm = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
    offs_cn = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def _pertensor_or_pertoken_smm_launcher(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape

    grid = lambda META: (
        triton.cdiv(M, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),
    )

    ACC_DTYPE = tl.float32 if a.is_floating_point() else tl.int32

    _M_NPO2 = triton.next_power_of_2(M)

    IS_PER_TOKEN_A = a_scale.numel() == M
    IS_PER_TOKEN_B = b_scale.numel() == N

    _pertensor_or_pertoken_smm_kernel[grid](
        c,
        a,
        b,
        a_scale,
        b_scale,
        bias,
        M,
        N,
        K,
        _M_NPO2,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACC_DTYPE=ACC_DTYPE,
        IS_PER_TOKEN_A=IS_PER_TOKEN_A,
        IS_PER_TOKEN_B=IS_PER_TOKEN_B,
    )

    return c


cutlass_scaled_mm_sm90_fp8 = _pertensor_or_pertoken_smm_launcher

cutlass_scaled_mm_sm90_int8 = _pertensor_or_pertoken_smm_launcher

cutlass_scaled_mm_blockwise_sm90_fp8 = _block_wise_128_smm_launcher


def dispatch_scaled_mm(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    fp8_func: Callable,
    int8_func: Optional[Callable],
    blockwise_func: Callable,
) -> None:
    assert a_scale.dtype == torch.float32, "a_scale must be float32"
    assert b_scale.dtype == torch.float32, "b_scale must be float32"

    if (a_scale.numel() == 1 or a_scale.numel() == a.size(0)) and (
        b_scale.numel() == 1 or b_scale.numel() == b.size(1)
    ):
        assert a_scale.is_contiguous(), "a_scale must be contiguous"
        assert b_scale.is_contiguous(), "b_scale must be contiguous"

        if a.dtype == torch.float8_e4m3fn:
            fp8_func(c, a, b, a_scale, b_scale, bias)
        else:
            assert a.dtype == torch.int8, f"Unsupported dtype: {a.dtype}"

            if int8_func is not None:
                int8_func(c, a, b, a_scale, b_scale, bias)
            else:
                raise RuntimeError(
                    f"Int8 not supported on SM{SM_VERSION_NUM}. "
                    f"Use FP8 quantization instead, or run on older arch (SM < 100)."
                )
    else:
        assert a_scale.dim() == 2, "a_scale must be 2D tensor for blockwise scaling"
        assert b_scale.dim() == 2, "b_scale must be 2D tensor for blockwise scaling"

        if SM_VERSION_NUM >= 90:
            assert a.size(0) == a_scale.size(0), (
                f"a_scale must have same first dimension as a: "
                f"a.shape[0]={a.size(0)}, a_scale.shape[0]={a_scale.size(0)}"
            )
            assert triton.cdiv(a.size(1), 128) == a_scale.size(1), (
                f"a_scale second dimension mismatch: "
                f"triton.cdiv({a.size(1)}, 128)={triton.cdiv(a.size(1), 128)} != "
                f"a_scale.shape[1]={a_scale.size(1)}"
            )

            assert triton.cdiv(b.size(0), 128) == b_scale.size(0), (
                f"b_scale first dimension mismatch: "
                f"triton.cdiv({b.size(0)}, 128)={triton.cdiv(b.size(0), 128)} != "
                f"b_scale.shape[0]={b_scale.size(0)}"
            )
            assert triton.cdiv(b.size(1), 128) == b_scale.size(1), (
                f"b_scale second dimension mismatch: "
                f"triton.cdiv({b.size(1)}, 128)={triton.cdiv(b.size(1), 128)} != "
                f"b_scale.shape[1]={b_scale.size(1)}"
            )

        assert bias is None, "Bias not yet supported for blockwise scaled_mm"

        blockwise_func(c, a, b, a_scale, b_scale)


def cutlass_scaled_mm_sm90(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> None:
    dispatch_scaled_mm(
        c=c,
        a=a,
        b=b,
        a_scale=a_scale,
        b_scale=b_scale,
        bias=bias,
        fp8_func=cutlass_scaled_mm_sm90_fp8,
        int8_func=cutlass_scaled_mm_sm90_int8,
        blockwise_func=cutlass_scaled_mm_blockwise_sm90_fp8,
    )


def cutlass_scaled_mm_sm120(*args, **kwargs):
    raise NotImplementedError("cutlass_scaled_mm_sm120 is not yet implemented. ")


def cutlass_scaled_mm_sm100(*args, **kwargs):
    raise NotImplementedError("cutlass_scaled_mm_sm100 is not yet implemented. ")


def cutlass_scaled_mm_sm89(*args, **kwargs):
    raise NotImplementedError("cutlass_scaled_mm_sm89 is not yet implemented. ")


def cutlass_scaled_mm_sm80(*args, **kwargs):
    raise NotImplementedError("cutlass_scaled_mm_sm80 is not yet implemented. ")


def cutlass_scaled_mm_sm75(*args, **kwargs):
    raise NotImplementedError("cutlass_scaled_mm_sm75 is not yet implemented. ")


def cutlass_scaled_mm(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert (
        a.dim() == 2 and b.dim() == 2 and c.dim() == 2
    ), "All inputs must be 2D tensors"

    assert c.size(0) == a.size(0), "Number of rows in c must equal number of rows in a"
    assert a.size(1) == b.size(
        0
    ), "Number of columns in a must equal number of rows in b"
    assert b.size(1) == c.size(
        1
    ), "Number of columns in b must equal number of columns in c"

    assert a.stride(1) == 1 and c.stride(1) == 1, "a and c must be row-major"

    assert b.stride(0) == 1, "b must be column-major"

    assert c.stride(0) % 16 == 0, "Row stride of c must be 16-byte aligned"
    assert b.stride(1) % 16 == 0, "Column stride of b must be 16-byte aligned"

    if bias is not None:
        assert bias.numel() == b.size(
            1
        ), f"Bias size {bias.numel()} must equal number of columns in b {b.size(1)}"
        assert bias.is_contiguous(), "Bias must be contiguous"
        assert bias.dim() == 1, "Bias must be a 1D tensor"

    if SM_VERSION_NUM >= 120:
        cutlass_scaled_mm_sm120(c, a, b, a_scale, b_scale, bias)

    elif SM_VERSION_NUM >= 100:
        cutlass_scaled_mm_sm100(c, a, b, a_scale, b_scale, bias)

    elif SM_VERSION_NUM >= 90:
        # Hopper
        cutlass_scaled_mm_sm90(c, a, b, a_scale, b_scale, bias)

    elif SM_VERSION_NUM >= 80:
        # Ampere
        cutlass_scaled_mm_sm80(c, a, b, a_scale, b_scale, bias)

    elif SM_VERSION_NUM >= 75:
        # Turing
        cutlass_scaled_mm_sm75(c, a, b, a_scale, b_scale, bias)
