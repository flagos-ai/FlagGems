import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext


@libentry()
@triton.jit
def mm_fp32_static_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    pid = ext.program_id(0)

    # --------------------------
    # match naming: num_pid_m, num_pid_n
    # --------------------------
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # reorder for L2
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)

    pid_m = group_id * GROUP_M + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --------------------------
    # match naming: offs_am, offs_bn, offs_k
    # --------------------------
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # contiguous aligned offsets (ram/rbn → offs_am/offs_bn)
    offs_am_cont = tl.max_contiguous(tl.multiple_of(offs_am % M, BLOCK_M), BLOCK_M)
    offs_bn_cont = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_N), BLOCK_N)

    # previous K multiple
    # prev_k_mult = prev_multiple_of(K, BLOCK_K)
    prev_k_mult = tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K

    # accumulator
    if IS_FP64:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --------------------------
    # main K loop
    # --------------------------
    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = start_k + offs_k

        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak)
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn)
        )

        if a.dtype != b.dtype:
            a = a.to(c_ptr.dtype.element_ty)
            b = b.to(c_ptr.dtype.element_ty)

        if IS_FP64:
            accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # --------------------------
    # loop peel
    # --------------------------
    rk = prev_k_mult + offs_k
    mask_k = rk < K

    if IS_FP64:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_k[:, None],
            other=0.0,
        )
    else:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_k[:, None],
        )

    if a.dtype != b.dtype:
        a = a.to(c_ptr.dtype.element_ty)
        b = b.to(c_ptr.dtype.element_ty)

    if IS_FP64:
        accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
    else:
        accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # cast to output dtype
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # --------------------------
    # rematerialize offsets for store
    # (match naming: offs_cm, offs_cn)
    # --------------------------
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    c_ptr = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_store = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    tl.store(c_ptr, accumulator, mask=mask_store)


_MM_FP32_STATIC_CONFIGS = {
    (60000, 512, 1024): (64, 64, 32, 1, 4),
    (140000, 512, 512): (64, 64, 32, 1, 4),
}

# Wide-vocabulary (145, 2560, 262208) projections use upstream MM.
# The static tile regressed on HIP/gfx936 in the full independent comparison.
_MM_SPECIALIZATION_CONFIGS = {
    (145, 10240, 2560): (64, 64, 64, 2, 8),
    (145, 2048, 2560): (32, 64, 64, 2, 4),
    (145, 2560, 1024): (32, 32, 128, 2, 4),
    (145, 2560, 10240): (128, 128, 64, 2, 4),
    (145, 2560, 2048): (32, 64, 64, 1, 4),
}


@libentry()
@triton.jit
def mm_bf16_staticall_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    pid = ext.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_am_cont = tl.max_contiguous(tl.multiple_of(offs_am % M, BLOCK_M), BLOCK_M)
    offs_bn_cont = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_N), BLOCK_N)
    prev_k_mult = tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K
    if IS_FP64:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = start_k + offs_k
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak)
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn)
        )
        if a.dtype != b.dtype:
            a = a.to(c_ptr.dtype.element_ty)
            b = b.to(c_ptr.dtype.element_ty)
        if IS_FP64:
            accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    rk = prev_k_mult + offs_k
    mask_k = rk < K
    if IS_FP64:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_k[:, None],
            other=0.0,
        )
    else:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_k[:, None],
        )
    if a.dtype != b.dtype:
        a = a.to(c_ptr.dtype.element_ty)
        b = b.to(c_ptr.dtype.element_ty)
    if IS_FP64:
        accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
    else:
        accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptr = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_store = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.store(c_ptr, accumulator, mask=mask_store)


@libentry()
@triton.jit
def mm_bf16_staticmasked_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    pid = ext.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_am < M
    mask_n = offs_bn < N
    offs_k = tl.arange(0, BLOCK_K)
    offs_am_cont = offs_am
    offs_bn_cont = offs_bn
    prev_k_mult = tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K
    if IS_FP64:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_k_mult, BLOCK_K):
        rk = start_k + offs_k
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_m[:, None],
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_n[None, :],
            other=0.0,
        )
        if a.dtype != b.dtype:
            a = a.to(c_ptr.dtype.element_ty)
            b = b.to(c_ptr.dtype.element_ty)
        if IS_FP64:
            accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    rk = prev_k_mult + offs_k
    mask_k = rk < K
    if IS_FP64:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        )
    else:
        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        )
    if a.dtype != b.dtype:
        a = a.to(c_ptr.dtype.element_ty)
        b = b.to(c_ptr.dtype.element_ty)
    if IS_FP64:
        accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
    else:
        accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
    accumulator = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptr = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_store = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.store(c_ptr, accumulator, mask=mask_store)


_MM_SPECIALIZATION_KERNELS = {
    (145, 10240, 2560): mm_bf16_staticmasked_kernel,
    (145, 2048, 2560): mm_bf16_staticall_kernel,
    (145, 2560, 1024): mm_bf16_staticmasked_kernel,
    (145, 2560, 10240): mm_bf16_staticmasked_kernel,
    (145, 2560, 2048): mm_bf16_staticall_kernel,
}

_MM_SPLIT_K_CONFIGS = {(145, 10240, 2560): 4}


@libentry()
@triton.jit
def mm_bf16_splitk_partial_kernel(
    a_ptr,
    b_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = ext.program_id(0)
    split = ext.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, K // SPLIT_K, BLOCK_K):
        kk = split * (K // SPLIT_K) + start_k + offsets_k
        left = tl.load(
            a_ptr + rows[:, None] * K + kk[None, :], mask=rows[:, None] < M, other=0.0
        )
        right = tl.load(
            b_ptr + kk[:, None] + cols[None, :] * K, mask=cols[None, :] < N, other=0.0
        )
        accumulator += tl.dot(left, right, out_dtype=tl.float32, allow_tf32=False)
    address = split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(
        partial_ptr + address,
        accumulator,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@libentry()
@triton.jit
def mm_bf16_splitk_reduce_kernel(
    partial_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < M * N
    value = tl.load(partial_ptr + index, mask=mask, other=0.0)
    for split in tl.static_range(1, SPLIT_K):
        value = value + tl.load(
            partial_ptr + split * M * N + index, mask=mask, other=0.0
        )
    tl.store(c_ptr + index, value.to(c_ptr.dtype.element_ty), mask=mask)


def try_linear_projection(a, b):
    # Only the original projection domains enter this optional specialization.
    # Returning None preserves the dependency's MM/FP64 dispatch everywhere else.
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return None
    m, k = a.shape
    n = b.shape[1]
    if a.device != b.device or a.device.type not in ("flagos", "cuda"):
        return None
    if a.stride() != (k, 1) or b.stride() != (1, k):
        return None
    bf16 = (
        a.dtype == b.dtype == torch.bfloat16 and (m, k, n) in _MM_SPECIALIZATION_CONFIGS
    )
    fp32 = (
        a.dtype == b.dtype == torch.float32
        and (m, k, n) in _MM_FP32_STATIC_CONFIGS
        and a.storage_offset() == b.storage_offset() == 0
    )
    if not (bf16 or fp32):
        return None
    device = a.device
    original_a = a
    original_b = b
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c_dtype = a.dtype
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    block_m, block_n, block_k, stages, warps = (64, 64, 32, 1, 4)
    launch_kernel = None
    config = _MM_SPECIALIZATION_CONFIGS.get((M, K, N))
    if (
        config is not None
        and a is original_a
        and (b is original_b)
        and (a.dtype is torch.bfloat16)
        and (b.dtype is torch.bfloat16)
        and (device == b.device)
        and (device.type in ("flagos", "cuda"))
        and (a.stride(0) == K)
        and (a.stride(1) == 1)
        and (b.stride(0) == 1)
        and (b.stride(1) == K)
    ):
        block_m, block_n, block_k, stages, warps = config
        launch_kernel = _MM_SPECIALIZATION_KERNELS[(M, K, N)]
        split_k = _MM_SPLIT_K_CONFIGS.get((M, K, N))
        if split_k is not None and K % (split_k * block_k) == 0:
            partial = torch.empty((split_k, M, N), device=device, dtype=torch.float32)
            with torch_device_fn.device(a.device):
                mm_bf16_splitk_partial_kernel[
                    triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k
                ](
                    a,
                    b,
                    partial,
                    M,
                    N,
                    K,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    GROUP_M=8,
                    SPLIT_K=split_k,
                    num_stages=stages,
                    num_warps=warps,
                    num_ldmatrixes=0,
                )
                mm_bf16_splitk_reduce_kernel[triton.cdiv(M * N, 256),](
                    partial,
                    c,
                    M,
                    N,
                    SPLIT_K=split_k,
                    BLOCK=256,
                    num_stages=1,
                    num_warps=4,
                )
            return c
    fp32_config = _MM_FP32_STATIC_CONFIGS.get((M, K, N))
    if (
        fp32_config is not None
        and a.dtype is torch.float32
        and b.dtype is torch.float32
        and device == b.device
        and device.type in ("flagos", "cuda")
        and a.stride() == (K, 1)
        and b.stride() == (1, K)
        and a.storage_offset() == 0
        and b.storage_offset() == 0
    ):
        block_m, block_n, block_k, stages, warps = fp32_config
        launch_kernel = mm_fp32_static_kernel
    with torch_device_fn.device(a.device):
        launch_kernel[grid](
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_stages=stages,
            num_warps=warps,
            num_ldmatrixes=0,
            IS_FP64=a.dtype == torch.float64,
        )
    return c


@libentry()
@triton.jit
def addmm_single_row_k1_kernel(
    A,
    B,
    Bias,
    C,
    N: tl.constexpr,
    B_STRIDE: tl.constexpr,
    I_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(A)
    weight = tl.load(B + index * B_STRIDE, index < N, other=0.0)
    bias = tl.load(Bias + index * I_STRIDE, index < N, other=0.0)
    value = x * weight + bias
    tl.store(C + index, value, index < N)


def try_linear_affine(bias, mat1, mat2, *, beta=1, alpha=1):
    M, K = mat1.shape
    N = mat2.shape[1]
    if (
        M == 1
        and K == 1
        and N > 0
        and type(alpha) in (int, float)
        and type(beta) in (int, float)
        and alpha == 1
        and beta == 1
        and mat1.dtype == torch.float32
        and mat2.dtype == torch.float32
        and bias.dtype == torch.float32
        and mat1.device == mat2.device == bias.device
        and mat1.device.type in ("flagos", "cuda")
        and bias.ndim == 1
        and bias.shape[0] == N
    ):
        result = torch.empty((1, N), device=mat1.device, dtype=mat1.dtype)
        with torch_device_fn.device(mat1.device):
            addmm_single_row_k1_kernel[(triton.cdiv(N, 256),)](
                mat1,
                mat2,
                bias,
                result,
                N,
                B_STRIDE=mat2.stride(1),
                I_STRIDE=bias.stride(0),
                BLOCK=256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return result

    return None
