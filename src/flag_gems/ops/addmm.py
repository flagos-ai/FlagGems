import logging

import paddle
import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import device_guard
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    # The transposed benchmark input has a different memory layout for B.
    # Keep its autotuned tile separate from the row-major case.
    key=["M", "N", "K", "stride_bk"],
    strategy=["align32", "align32", "align32", "default"],
    warmup=5,
    rep=10,
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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm_splitk"),
    key=["M", "N", "K", "stride_bk"],
    strategy=["align32", "align32", "align32", "default"],
    # The kernel accumulates into c_ptr, which arrives pre-seeded with beta * bias.
    # Without this the autotuner's benchmark runs would each add another product into
    # it and the first call for a new key would return garbage.
    restore_value=["c_ptr"],
    warmup=5,
    rep=10,
)
@triton.jit(do_not_specialize=["alpha"])
def addmm_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Tile over K as well, accumulating atomically into a pre-seeded output.

    The kernel above only tiles M and N, so a shape like M=8, N=512 fills 4 programs
    on a 132-SM part and runs at a fraction of paddle's GEMV-style kernel. Here each
    program walks every SPLIT_K-th K tile, which multiplies the program count by
    SPLIT_K. `c_ptr` must already hold beta * bias.
    """
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    pid_k = tle.program_id(2)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    step = BLOCK_SIZE_K * SPLIT_K
    for k in range(0, tl.cdiv(K - pid_k * BLOCK_SIZE_K, step)):
        k_remaining = K - pid_k * BLOCK_SIZE_K - k * step
        k_mask = tl.arange(0, BLOCK_SIZE_K) < k_remaining
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += step * stride_ak
        b_ptrs += step * stride_bk

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    c = (accumulator * alpha).to(c_ptr.dtype.element_ty)
    tl.atomic_add(c_ptrs, c, mask=c_mask)


def addmm_paddle(bias, mat1, mat2, beta=1, alpha=1):# To maintain consistency in the number of parameters for custom operators and _C_ops.func
    return addmm(bias, mat1, mat2, beta=beta, alpha=alpha)

def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    # logger.debug(
    #     "GEMS ADDMM, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
    #     "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
    #     M,
    #     N,
    #     K,
    #     mat1.stride(0) == 1,
    #     mat2.stride(0) == 1,
    #     bias.stride(0) == 1,
    # )
    mat1 = mat1.contiguous()
    # mat2 = mat2.contiguous()
    # Shapes whose tile grid cannot fill the device: at M <= 128 and N <= 4096 the
    # single-pass kernel launches at most a few dozen programs on 132 SMs and lands at
    # 0.2-0.7x of paddle, which dispatches a GEMV-like kernel there. Splitting the K
    # loop instead multiplies the program count; measured 0.80-1.81x on those shapes.
    # Larger shapes already saturate the device and keep the single-pass path, where
    # they are at ~1.0x -- splitting them costs an extra pass over the output.
    if M <= 128 and N <= 4096 and K >= 2048:
        if beta == 0:
            out = paddle._C_ops.full([M, N], 0.0, mat1.dtype, mat1.place)
        else:
            out = paddle._C_ops.scale(bias.broadcast_to([M, N]), float(beta), 0.0, True)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
            META["SPLIT_K"],
        )
        with device_guard(mat1):
            addmm_splitk_kernel[grid](
                mat1,
                mat2,
                out,
                alpha,
                M,
                N,
                K,
                mat1.stride(0),
                mat1.stride(1),
                mat2.stride(0),
                mat2.stride(1),
                out.stride(0),
                out.stride(1),
            )
        return out

    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    # with torch_device_fn.device(mat1.device):
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


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    logger.debug(
        "GEMS ADDMM_OUT, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
        M,
        N,
        K,
        mat1.stride(0) == 1,
        mat2.stride(0) == 1,
        bias.stride(0) == 1,
    )
    mat1 = mat1.contiguous()
    bias = bias.broadcast_to(out.shape)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    with device_guard(mat1):
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
