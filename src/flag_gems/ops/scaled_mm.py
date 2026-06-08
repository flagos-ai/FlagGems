import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.fused.cutlass_scaled_mm import cutlass_scaled_mm as _csmm
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

GROUP_M = 8
SCALAR_SCALE = 0
VECTOR_SCALE = 1
ASCEND_ALIGNED_BLOCK = 128
ASCEND_ALIGNED_KERNEL_BLOCK = 64
ASCEND_ALIGNED_MIN_VOLUME = 512 * 512 * 512


def _heur_even_k(args):
    return args["K"] % args["BLOCK_K"] == 0


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("scaled_mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=2,
    rep=4,
)
@triton.heuristics({"EVEN_K": _heur_even_k})
@triton.jit
def scaled_mm_kernel(
    A,
    B,
    ScaleA,
    ScaleB,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    SCALE_A_MODE: tl.constexpr,
    SCALE_B_MODE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = offs_m.to(tl.int64)
    offs_n = offs_n.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                other=0.0,
            )
        acc += tl.dot(a, b, out_dtype=ACC_DTYPE, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc.to(tl.float32)

    if SCALE_A_MODE == 0:
        scale_a = tl.full((BLOCK_M,), tl.load(ScaleA), dtype=tl.float32)
    else:
        scale_a = tl.load(ScaleA + offs_m, mask=offs_m < M, other=0.0)

    if SCALE_B_MODE == 0:
        scale_b = tl.full((BLOCK_N,), tl.load(ScaleB), dtype=tl.float32)
    else:
        scale_b = tl.load(ScaleB + offs_n, mask=offs_n < N, other=0.0)

    acc = acc * scale_a[:, None] * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@libentry()
@triton.jit
def scaled_mm_aligned_kernel(
    A,
    B,
    ScaleA,
    ScaleB,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    SCALE_A_MODE: tl.constexpr,
    SCALE_B_MODE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tle.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = offs_m.to(tl.int64)
    offs_n = offs_n.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, out_dtype=ACC_DTYPE, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc.to(tl.float32)

    if SCALE_A_MODE == 0:
        scale_a = tl.full((BLOCK_M,), tl.load(ScaleA), dtype=tl.float32)
    else:
        scale_a = tl.load(ScaleA + offs_m)

    if SCALE_B_MODE == 0:
        scale_b = tl.full((BLOCK_N,), tl.load(ScaleB), dtype=tl.float32)
    else:
        scale_b = tl.load(ScaleB + offs_n)

    acc = acc * scale_a[:, None] * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(Bias + offs_n)
        acc += bias[None, :]

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def _resolve_out_dtype(self, out_dtype, out=None):
    if out_dtype is not None:
        if out is not None and out.dtype != out_dtype:
            raise RuntimeError(
                "out_dtype must be the same as the dtype of the provided out tensor"
            )
        return out_dtype
    if out is not None:
        return out.dtype
    return self.dtype


def _normalize_scale(scale, expected_size, *, is_left_scale):
    if scale.numel() == 1:
        return scale.reshape(1).contiguous(), SCALAR_SCALE

    valid_vector = scale.ndim == 1 and scale.shape[0] == expected_size
    if is_left_scale:
        valid_vector = valid_vector or (
            scale.ndim == 2 and scale.shape == (expected_size, 1)
        )
    else:
        valid_vector = valid_vector or (
            scale.ndim == 2 and scale.shape == (1, expected_size)
        )

    if valid_vector:
        return scale.reshape(expected_size).contiguous(), VECTOR_SCALE

    scale_name = "scale_a" if is_left_scale else "scale_b"
    expected_shape = (
        f"({expected_size}, 1)" if is_left_scale else f"(1, {expected_size})"
    )
    raise RuntimeError(
        f"{scale_name} must be a scalar tensor or have shape {expected_shape}"
    )


def _normalize_bias(bias, cols):
    if bias is None:
        return None
    if bias.numel() != cols:
        raise RuntimeError(f"Bias must be size {cols} but got {bias.numel()}")
    return bias.reshape(cols).contiguous()


def _check_inputs(self, mat2):
    if self.ndim != 2:
        raise RuntimeError("self must be a matrix")
    if mat2.ndim != 2:
        raise RuntimeError("mat2 must be a matrix")
    if self.shape[1] != mat2.shape[0]:
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied ({self.shape[0]}x{self.shape[1]} "
            f"and {mat2.shape[0]}x{mat2.shape[1]})"
        )
    if self.dtype != mat2.dtype:
        raise RuntimeError(
            f"self and mat2 must have the same dtype, but got {self.dtype} and {mat2.dtype}"
        )


def _maybe_make_contiguous_for_kernel(self, mat2):
    if self.stride(0) > 1 and self.stride(1) > 1:
        self = self.contiguous()
    if mat2.stride(0) > 1 and mat2.stride(1) > 1:
        mat2 = mat2.contiguous()
    return self, mat2


def _can_use_cutlass_scaled_mm(self, mat2, scale_a, scale_b, bias, out):
    if self.device.type != "cuda":
        return False
    is_fp8 = hasattr(torch, "float8_e4m3fn") and self.dtype == torch.float8_e4m3fn
    if not (is_fp8 or self.dtype == torch.int8):
        return False
    if self.dtype != mat2.dtype:
        return False
    major, minor = torch.cuda.get_device_capability(self.device)
    sm_version_num = major * 10 + minor
    if not (90 <= sm_version_num < 100):
        return False
    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        return False
    if scale_a.numel() not in (1, self.shape[0]):
        return False
    if scale_b.numel() not in (1, mat2.shape[1]):
        return False
    if not scale_a.is_contiguous() or not scale_b.is_contiguous():
        return False
    if self.stride(1) != 1 or out.stride(1) != 1:
        return False
    if mat2.stride(0) != 1:
        return False
    if out.stride(0) % 16 != 0 or mat2.stride(1) % 16 != 0:
        return False
    if bias is not None and (bias.ndim != 1 or not bias.is_contiguous()):
        return False
    return True


def _can_use_ascend_aligned_scaled_mm(self, mat2, out):
    if self.device.type != "npu" or runtime.device.vendor_name != "ascend":
        return False
    if not self.is_floating_point():
        return False
    M, K = self.shape
    _, N = mat2.shape
    return (
        M * N * K >= ASCEND_ALIGNED_MIN_VOLUME
        and M % ASCEND_ALIGNED_BLOCK == 0
        and N % ASCEND_ALIGNED_BLOCK == 0
        and K % ASCEND_ALIGNED_BLOCK == 0
        and self.stride(1) == 1
        and mat2.stride(1) == 1
        and out.stride(1) == 1
    )


def _scaled_mm_impl(
    self,
    mat2,
    scale_a,
    scale_b,
    bias,
    out_dtype,
    out,
):
    _check_inputs(self, mat2)
    M, K = self.shape
    _, N = mat2.shape

    output_dtype = _resolve_out_dtype(self, out_dtype, out)
    if out is None:
        out = torch.empty((M, N), dtype=output_dtype, device=self.device)
    else:
        if out.shape != (M, N):
            raise RuntimeError("Incompatible output shape")

    scale_a, scale_a_mode = _normalize_scale(scale_a, M, is_left_scale=True)
    scale_b, scale_b_mode = _normalize_scale(scale_b, N, is_left_scale=False)
    bias = _normalize_bias(bias, N)

    if M == 0 or N == 0:
        return out

    if _can_use_cutlass_scaled_mm(self, mat2, scale_a, scale_b, bias, out):
        with torch_device_fn.device(self.device):
            _csmm(out, self, mat2, scale_a, scale_b, bias)
        return out

    self, mat2 = _maybe_make_contiguous_for_kernel(self, mat2)
    acc_dtype = tl.float32 if self.is_floating_point() else tl.int32
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(self.device):
        if _can_use_ascend_aligned_scaled_mm(self, mat2, out):
            block = ASCEND_ALIGNED_KERNEL_BLOCK
            aligned_grid = (triton.cdiv(M, block) * triton.cdiv(N, block),)
            scaled_mm_aligned_kernel[aligned_grid](
                self,
                mat2,
                scale_a,
                scale_b,
                bias,
                out,
                M,
                N,
                K,
                self.stride(0),
                self.stride(1),
                mat2.stride(0),
                mat2.stride(1),
                out.stride(0),
                out.stride(1),
                ACC_DTYPE=acc_dtype,
                SCALE_A_MODE=scale_a_mode,
                SCALE_B_MODE=scale_b_mode,
                HAS_BIAS=bias is not None,
                BLOCK_M=block,
                BLOCK_N=block,
                BLOCK_K=block,
                GROUP_M=GROUP_M,
            )
        else:
            scaled_mm_kernel[grid](
                self,
                mat2,
                scale_a,
                scale_b,
                bias,
                out,
                M,
                N,
                K,
                self.stride(0),
                self.stride(1),
                mat2.stride(0),
                mat2.stride(1),
                out.stride(0),
                out.stride(1),
                ACC_DTYPE=acc_dtype,
                SCALE_A_MODE=scale_a_mode,
                SCALE_B_MODE=scale_b_mode,
                HAS_BIAS=bias is not None,
                GROUP_M=GROUP_M,
            )
    return out


def scaled_mm(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    logger.debug("GEMS SCALED_MM")
    return _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, out_dtype, None)


def scaled_mm_out(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out,
):
    logger.debug("GEMS SCALED_MM_OUT")
    return _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, out_dtype, out)
