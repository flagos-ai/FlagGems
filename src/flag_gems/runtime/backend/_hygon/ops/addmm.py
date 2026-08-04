import logging
from numbers import Number

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


def _scalar_eq(value, target):
    return isinstance(value, Number) and value == target


def _validate_addmm_shapes(bias, mat1, mat2):
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"


def _is_col_major(tensor):
    return tensor.dim() >= 1 and tensor.stride(0) == 1


def _finish_bias_only(out, bias, beta):
    bias = bias.broadcast_to(out.shape)
    if _scalar_eq(beta, 0):
        return out.zero_()
    if _scalar_eq(beta, 1):
        return out.copy_(bias)
    return torch.mul(bias, beta, out=out)


def _allow_tf32(mat1, mat2):
    if mat1.dtype != torch.float32 or mat2.dtype != torch.float32:
        return False
    if mat1.device.type != "cuda":
        return False
    cuda_backend = getattr(torch.backends, "cuda", None)
    return bool(cuda_backend and cuda_backend.matmul.allow_tf32)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
    flagtune_op_name="addmm",
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
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
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
    ALPHA_ONE: tl.constexpr = False,
    BETA_ZERO: tl.constexpr = False,
    BETA_ONE: tl.constexpr = False,
    BIAS_SCALAR: tl.constexpr = False,
    BIAS_ROW: tl.constexpr = False,
    BIAS_COL: tl.constexpr = False,
    B_CONTIGUOUS: tl.constexpr = False,
    EVEN_M: tl.constexpr = False,
    EVEN_N: tl.constexpr = False,
    EVEN_K: tl.constexpr = False,
    ALLOW_TF32: tl.constexpr = False,
    IS_FP64: tl.constexpr = False,
):
    pid = ext.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if EVEN_M:
        offs_am_c = tl.max_contiguous(
            tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M
        )
    else:
        offs_am_c = tl.max_contiguous(
            tl.multiple_of(offs_am % M, BLOCK_SIZE_M), BLOCK_SIZE_M
        )
    if EVEN_N:
        offs_bn_c = tl.max_contiguous(
            tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N
        )
    else:
        offs_bn_c = tl.max_contiguous(
            tl.multiple_of(offs_bn % N, BLOCK_SIZE_N), BLOCK_SIZE_N
        )

    # A is always row-major (mat1.contiguous() is guaranteed by _launch_addmm)
    a_ptrs = a_ptr + offs_am_c[:, None] * K + offs_k[None, :]
    if B_CONTIGUOUS:
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_bn_c[None, :]
        b_step = BLOCK_SIZE_K * N
    else:
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn_c[None, :] * stride_bn
        b_step = BLOCK_SIZE_K * stride_bk

    if IS_FP64:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if EVEN_K:
        loop_end = K
    else:
        loop_end = tl.cdiv(K, BLOCK_SIZE_K) * BLOCK_SIZE_K - BLOCK_SIZE_K
    for k in range(0, loop_end, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if IS_FP64:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        accumulator += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += b_step

    if not EVEN_K:
        rk = loop_end + offs_k
        mask_k = rk < K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        if IS_FP64:
            a = a.to(tl.float32)
            b = b.to(tl.float32)
        accumulator += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    if not ALPHA_ONE:
        accumulator *= alpha

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if BETA_ZERO:
        c = accumulator
    else:
        if BIAS_SCALAR:
            bias = tl.load(i_ptr)
        elif BIAS_ROW:
            bias_ptrs = i_ptr + stride_in * offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)[None, :]
        elif BIAS_COL:
            bias_ptrs = i_ptr + stride_im * offs_cm
            bias = tl.load(bias_ptrs, mask=offs_cm < M, other=0.0)[:, None]
        else:
            i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
            if EVEN_M and EVEN_N:
                bias = tl.load(i_ptrs)
            else:
                bias = tl.load(i_ptrs, mask=c_mask, other=0.0)
        if BETA_ONE:
            c = accumulator + bias
        else:
            c = accumulator + bias * beta
        c = c.to(bias.dtype)

    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=c_mask)


def _launch_addmm(bias, mat1, mat2, out, *, beta=1, alpha=1):
    M, K = mat1.shape
    _, N = mat2.shape

    if M == 0 or N == 0:
        return out

    if K == 0 or _scalar_eq(alpha, 0):
        return _finish_bias_only(out, bias, beta)

    logger.debug(
        "GEMS ADDMM, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
        M,
        N,
        K,
        mat1.stride(0) == 1,
        _is_col_major(mat2),
        _is_col_major(bias),
    )

    mat1 = mat1.contiguous()
    beta_zero = _scalar_eq(beta, 0)
    bias_scalar = False
    bias_row = False
    bias_col = False
    if not beta_zero:
        bias = bias.broadcast_to(out.shape)
        stride_im = bias.stride(0)
        stride_in = bias.stride(1)
        bias_scalar = stride_im == 0 and stride_in == 0
        bias_row = stride_im == 0 and stride_in != 0
        bias_col = stride_im != 0 and stride_in == 0
    else:
        stride_im = 0
        stride_in = 0

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
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
            mat2.stride(0),
            mat2.stride(1),
            stride_im,
            stride_in,
            out.stride(0),
            out.stride(1),
            ALPHA_ONE=_scalar_eq(alpha, 1),
            BETA_ZERO=beta_zero,
            BETA_ONE=_scalar_eq(beta, 1),
            BIAS_SCALAR=bias_scalar,
            BIAS_ROW=bias_row,
            BIAS_COL=bias_col,
            B_CONTIGUOUS=mat2.stride(0) == N and mat2.stride(1) == 1,
            ALLOW_TF32=_allow_tf32(mat1, mat2),
            IS_FP64=mat1.dtype == torch.float64,
        )
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    _validate_addmm_shapes(bias, mat1, mat2)
    M = mat1.shape[0]
    N = mat2.shape[1]
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    return _launch_addmm(bias, mat1, mat2, out, beta=beta, alpha=alpha)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    _validate_addmm_shapes(bias, mat1, mat2)
    M = mat1.shape[0]
    N = mat2.shape[1]
    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"
    return _launch_addmm(bias, mat1, mat2, out, beta=beta, alpha=alpha)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    logger.debug("GEMS ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    logger.debug("GEMS ADDMM_DTYPE_OUT")
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got {mat1.dtype} and {mat2.dtype}"
        )
    if out.dtype != out_dtype:
        raise RuntimeError(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    if not (
        out_dtype == mat1.dtype
        or (
            out_dtype == torch.float32 and mat1.dtype in (torch.float16, torch.bfloat16)
        )
    ):
        raise RuntimeError(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs"
        )
    if bias.dtype != out_dtype and bias.dtype != mat1.dtype:
        raise RuntimeError("self dtype must match either out_dtype or mat1 dtype")

    bias_c = bias if _scalar_eq(beta, 0) else bias.to(out_dtype)
    return addmm_out(bias_c, mat1, mat2, beta=beta, alpha=alpha, out=out)
