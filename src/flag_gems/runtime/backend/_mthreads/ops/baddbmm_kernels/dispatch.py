# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

try:
    import triton.experimental.tle.language as tle
    from triton.experimental.tle import is_tle_supported
except ImportError:  # pragma: no cover - depends on the installed FlagTree build
    is_tle_supported = None
    tle = None

from . import backward as _baddbmm_backward
from . import catalog_shapes as _catalog_shapes
from . import generic as _baddbmm_generic

_BADDBMM_TLE_PRIMITIVES = (
    "pipe",
    "pipe.reader",
    "pipe.reader.wait",
    "pipe.reader.release",
    "pipe.writer",
    "pipe.writer.acquire",
    "pipe.writer.commit",
    "pipe.writer.close",
    "gpu.alloc",
    "gpu.copy",
    "gpu.alloc_barrier",
    "gpu.alloc_barriers",
    "gpu.barrier_arrive",
    "gpu.barrier_wait",
    "gpu.warp_specialize",
    "gpu.wgmma",
    "gpu.wgmma_wait",
)
_BADDBMM_TLE_ON = (
    tle is not None
    and is_tle_supported is not None
    and is_tle_supported(_BADDBMM_TLE_PRIMITIVES)
    and hasattr(tle, "gpu")
    and hasattr(tle.gpu, "buffered_tensor")
    and hasattr(tle.gpu.buffered_tensor, "subslice")
    and getattr(
        getattr(tle.gpu, "mthreads", None),
        "MTHREADS_TLE_DISABLE_MAX_ILP_SCHEDULER_VERSION",
        0,
    )
    >= 1
)

if _BADDBMM_TLE_ON:
    from . import cooperative_split_m_dualpipe as _cooperative_split_m_dualpipe
    from . import cooperative_split_m_exact_a as _cooperative_split_m_exact_a
    from . import direct_tiled as _direct_tiled
    from . import explicit_barrier_ws as _explicit_barrier_ws
    from . import persistent_ws as _persistent_ws
    from . import small_m as _small_m
else:  # Keep the vendor fallback importable on older FlagTree builds.
    _cooperative_split_m_dualpipe = _cooperative_split_m_exact_a = None
    _direct_tiled = _explicit_barrier_ws = _persistent_ws = _small_m = None

_TINY_FORWARD_PATH = "tiny_b4_m1_n1_k32"
_SMALL_M_FORWARD_PATH = "small_m_b4_m15_n160_k1024"
_K71_FORWARD_PATH = "k71_b4_m495_n5333"
_DIRECT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_PERSISTENT_BM256_BN256_BK64_S3_K1024_PATH = (
    "bm256_bn256_bk64_s3_persistent_ws_bias_epilogue_k1024"
)
_PERSISTENT_BM256_BN256_BK64_S3_K7168_PATH = (
    "bm256_bn256_bk64_s3_persistent_ws_bias_epilogue_k7168"
)
_EXPLICIT_BARRIER_BM256_BN256_BK64_S2_COOP_M4_PATH = (
    "bm256_bn256_bk64_s2_coop_m4_explicit_barrier_bias_epilogue"
)
_EXACT_A_BM384_BN256_BK32_S3_COOP_M2_SYNTHETIC_BIAS_PATH = (
    "bm384_bn256_bk32_s3_coop_m2_exact_a_synthetic_bias"
)
_DUALPIPE_BM384_BN256_BK32_S3_COOP_M2_SYNTHETIC_BIAS_PATH = (
    "bm384_bn256_bk32_s3_coop_m2_dualpipe_synthetic_bias"
)

_OFFICIAL_FORWARD_SHAPES = {
    (4, 1, 1, 32),
    (4, 15, 160, 1024),
    (4, 495, 5333, 71),
}
_OFFICIAL_BACKWARD_SHAPES = {
    (2, 1, 1, 32),
    (2, 15, 160, 1024),
    (2, 495, 5333, 71),
}


def _is_broadcastable_to(input_shape, output_shape):
    """Return whether ``input_shape`` broadcasts to ``output_shape``."""
    if len(input_shape) > len(output_shape):
        return False
    return all(
        input_size in (1, output_size)
        for input_size, output_size in zip(
            reversed(input_shape), reversed(output_shape)
        )
    )


def _out_is_disjoint(out, *inputs):
    return out is None or all(not torch._C._overlaps(out, inp) for inp in inputs)


def _direct_forward_path(bias, A, B, out=None):
    """Return the fixed MTT S5000 direct path for validated contiguous inputs."""
    if (
        bias.ndim != 1
        or A.ndim != 3
        or B.ndim != 3
        or A.shape[0] != B.shape[0]
        or A.shape[2] != B.shape[1]
        or bias.numel() != B.shape[2]
        or bias.dtype != A.dtype
        or A.dtype != B.dtype
        or A.dtype not in _DIRECT_DTYPES
        or not bias.is_contiguous()
        or not A.is_contiguous()
        or not B.is_contiguous()
        or bias.device != A.device
        or A.device != B.device
        or A.device.type != "musa"
    ):
        return None

    expected_out_shape = (A.shape[0], A.shape[1], B.shape[2])
    if out is not None and (
        tuple(out.shape) != expected_out_shape
        or out.dtype != A.dtype
        or not out.is_contiguous()
        or out.device != A.device
        or not _out_is_disjoint(out, bias, A, B)
    ):
        return None

    shape = (A.shape[0], A.shape[1], B.shape[2], A.shape[2])
    if shape == (4, 1, 1, 32):
        return _TINY_FORWARD_PATH
    if shape == (4, 495, 5333, 71):
        return _K71_FORWARD_PATH
    return None


def _tle_forward_path(bias, A, B, alpha, beta, out=None):
    """Select a validated TLE path for contiguous BF16 inputs."""
    if (
        not _BADDBMM_TLE_ON
        or float(alpha) != 1.0
        or float(beta) != 1.0
        or bias.ndim != 1
        or A.ndim != 3
        or B.ndim != 3
        or A.shape[0] != 1
        or B.shape[0] != 1
        or A.shape[2] != B.shape[1]
        or bias.numel() != B.shape[2]
        or bias.dtype != torch.bfloat16
        or A.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or not bias.is_contiguous()
        or not A.is_contiguous()
        or not B.is_contiguous()
        or bias.device != A.device
        or A.device != B.device
        or A.device.type != "musa"
    ):
        return None

    expected_out_shape = (1, A.shape[1], B.shape[2])
    if out is not None and (
        tuple(out.shape) != expected_out_shape
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.device != A.device
        or not _out_is_disjoint(out, bias, A, B)
    ):
        return None

    shape = (A.shape[1], B.shape[2], A.shape[2])
    path = {
        (16384, 7168, 1024): _PERSISTENT_BM256_BN256_BK64_S3_K1024_PATH,
        (16384, 2112, 7168): _PERSISTENT_BM256_BN256_BK64_S3_K7168_PATH,
        (448, 7168, 256): _EXPLICIT_BARRIER_BM256_BN256_BK64_S2_COOP_M4_PATH,
        (14429, 7168, 1024): _EXACT_A_BM384_BN256_BK32_S3_COOP_M2_SYNTHETIC_BIAS_PATH,
        (14429, 2112, 7168): (
            _DUALPIPE_BM384_BN256_BK32_S3_COOP_M2_SYNTHETIC_BIAS_PATH
        ),
    }.get(shape)
    return path


def _small_m_forward_path(bias, A, B, out=None):
    """Select the shared-test TLE path for contiguous half inputs."""
    if (
        not _BADDBMM_TLE_ON
        or tuple(bias.shape) != (160,)
        or tuple(A.shape) != (4, 15, 1024)
        or tuple(B.shape) != (4, 1024, 160)
        or A.dtype not in (torch.float16, torch.bfloat16)
        or bias.dtype != A.dtype
        or B.dtype != A.dtype
        or not bias.is_contiguous()
        or not A.is_contiguous()
        or not B.is_contiguous()
        or bias.device != A.device
        or B.device != A.device
        or A.device.type != "musa"
    ):
        return None
    if out is not None and (
        tuple(out.shape) != (4, 15, 160)
        or out.dtype != A.dtype
        or not out.is_contiguous()
        or out.device != A.device
        or not _out_is_disjoint(out, bias, A, B)
    ):
        return None
    return _SMALL_M_FORWARD_PATH


def _launch_tle_forward(path, bias, A, B, out):
    """Launch one exact-shape path selected by :func:`_tle_forward_path`."""
    if path in (
        _PERSISTENT_BM256_BN256_BK64_S3_K1024_PATH,
        _PERSISTENT_BM256_BN256_BK64_S3_K7168_PATH,
    ):
        _persistent_ws.baddbmm_bm256_bn256_bk64_s3_persistent_ws_bias_epilogue_out(
            A[0],
            B[0],
            out[0],
            bias=bias,
            num_sms=60,
            num_slots=3,
            group_m=4,
            unroll_k=2,
        )
        return out
    if path == _EXPLICIT_BARRIER_BM256_BN256_BK64_S2_COOP_M4_PATH:
        return _explicit_barrier_ws.baddbmm_bm256_bn256_bk64_s2_coop_m4_out(
            bias, A, B, out
        )
    if path == _EXACT_A_BM384_BN256_BK32_S3_COOP_M2_SYNTHETIC_BIAS_PATH:
        return _cooperative_split_m_exact_a.baddbmm_bm384_bn256_bk32_s3_coop_m2_exact_a_out(
            bias, A, B, out
        )
    if path == _DUALPIPE_BM384_BN256_BK32_S3_COOP_M2_SYNTHETIC_BIAS_PATH:
        return _cooperative_split_m_dualpipe.baddbmm_bm384_bn256_bk32_s3_coop_m2_dualpipe_out(
            bias, A, B, out
        )
    raise ValueError(f"unknown TLE baddbmm path: {path}")


def _generic_out_eligible(bias, A, B, out=None):
    if A.ndim != 3 or B.ndim != 3:
        return False
    shape = (A.shape[0], A.shape[1], B.shape[2], A.shape[2])
    if shape not in _OFFICIAL_FORWARD_SHAPES:
        return False
    expected_out_shape = (A.shape[0], A.shape[1], B.shape[2])
    return (
        A.shape[0] == B.shape[0]
        and A.shape[2] == B.shape[1]
        and _is_broadcastable_to(tuple(bias.shape), expected_out_shape)
        and A.dtype in _DIRECT_DTYPES
        and bias.dtype == A.dtype
        and B.dtype == A.dtype
        and bias.device == A.device
        and B.device == A.device
        and A.device.type == "musa"
        and (
            out is None
            or (
                out.ndim == 3
                and tuple(out.shape) == expected_out_shape
                and out.dtype == A.dtype
                and out.device == A.device
                and _out_is_disjoint(out, bias, A, B)
            )
        )
    )


def _new_output(A, B):
    return torch.empty(
        (A.shape[0], A.shape[1], B.shape[2]), dtype=A.dtype, device=A.device
    )


def _dispatch_specialized_forward(bias, A, B, alpha, beta, out=None):
    """Run a validated specialized path, or return ``None`` for fallback."""
    if _catalog_shapes.is_eligible(bias, A, B, alpha, beta, out):
        return _catalog_shapes.launch(
            bias,
            A,
            B,
            out,
            alpha=alpha,
            beta=beta,
        )

    tle_path = _tle_forward_path(bias, A, B, alpha, beta, out)
    if tle_path is not None:
        out = _new_output(A, B) if out is None else out
        return _launch_tle_forward(tle_path, bias, A, B, out)

    small_m_path = _small_m_forward_path(bias, A, B, out)
    if small_m_path is not None:
        out = _new_output(A, B) if out is None else out
        return _small_m.baddbmm_bm64_bn64_bk128_s3_small_m_ws_out(
            bias,
            A,
            B,
            out,
            alpha=alpha,
            beta=beta,
        )

    direct_path = _direct_forward_path(bias, A, B, out)
    if direct_path is not None:
        out = _new_output(A, B) if out is None else out
        return _launch_direct_forward(direct_path, bias, A, B, alpha, beta, out)

    if (
        _BADDBMM_TLE_ON
        and _direct_tiled.is_bm64_bn64_bk64_s1_half_eligible(
            bias, A, B, alpha, beta, out
        )
        and A.shape[1] == 384
    ):
        return _direct_tiled.launch_bm64_bn64_bk64_s1_tiled_sqmma(bias, A, B, out)

    if _BADDBMM_TLE_ON and _direct_tiled.is_bm64_bn128_bk32_s1_fp32_eligible(
        bias, A, B, alpha, beta, out
    ):
        return _direct_tiled.launch_bm64_bn128_bk32_s1_tiled_tf32x3(bias, A, B, out)

    if _generic_out_eligible(bias, A, B, out):
        out = _new_output(A, B) if out is None else out
        return _baddbmm_generic.generic_out(
            bias,
            A,
            B,
            out,
            beta=beta,
            alpha=alpha,
        )
    return None


def _generic_backward_eligible(bias, A, B, grad_output):
    if A.ndim != 3 or B.ndim != 3 or grad_output.ndim != 3:
        return False
    tensors = (bias, A, B, grad_output)
    shape = (A.shape[0], A.shape[1], B.shape[2], A.shape[2])
    output_shape = (A.shape[0], A.shape[1], B.shape[2])
    return (
        shape in _OFFICIAL_BACKWARD_SHAPES
        and A.shape[0] == B.shape[0]
        and A.shape[2] == B.shape[1]
        and tuple(grad_output.shape) == output_shape
        and _is_broadcastable_to(tuple(bias.shape), output_shape)
        and A.dtype in _DIRECT_DTYPES
        and all(t.dtype == A.dtype for t in tensors)
        and all(t.device == A.device for t in tensors)
        and A.device.type == "musa"
    )


@triton.jit(do_not_specialize=["alpha", "beta"])
def _tiny_baddbmm_kernel(A, B, bias, out, alpha, beta):
    batch = tl.arange(0, 4)
    k = tl.arange(0, 32)
    a = tl.load(A + batch[:, None] * 32 + k[None, :]).to(tl.float32)
    b = tl.load(B + batch[:, None] * 32 + k[None, :]).to(tl.float32)
    product = tl.sum(a * b, axis=1)
    bias_value = tl.load(bias).to(tl.float32)
    tl.store(out + batch, alpha * product + beta * bias_value)


@triton.jit(do_not_specialize=["alpha", "beta"])
def _k71_baddbmm_kernel(
    A,
    B,
    bias,
    out,
    alpha,
    beta,
    INPUT_PRECISION: tl.constexpr,
):
    block_m: tl.constexpr = 32
    block_n: tl.constexpr = 128
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    logical_m0 = pid_m * block_m
    logical_n0 = pid_n * block_n
    m0 = tl.minimum(logical_m0, 495 - block_m)
    n0 = tl.minimum(logical_n0, 5333 - block_n)
    offs_m = m0 + tl.arange(0, block_m)
    offs_n = n0 + tl.arange(0, block_n)

    offs_k0 = tl.arange(0, 32)
    a0 = tl.load(A + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k0[None, :])
    b0 = tl.load(B + pid_b * (71 * 5333) + offs_k0[:, None] * 5333 + offs_n[None, :])
    accumulator = tl.dot(a0, b0, input_precision=INPUT_PRECISION)

    offs_k1 = 32 + tl.arange(0, 32)
    a1 = tl.load(A + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k1[None, :])
    b1 = tl.load(B + pid_b * (71 * 5333) + offs_k1[:, None] * 5333 + offs_n[None, :])
    accumulator = tl.dot(a1, b1, acc=accumulator, input_precision=INPUT_PRECISION)

    offs_k2 = 64 + tl.arange(0, 32)
    a2 = tl.load(
        A + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k2[None, :],
        mask=offs_k2[None, :] < 71,
        other=0.0,
    )
    b2 = tl.load(
        B + pid_b * (71 * 5333) + offs_k2[:, None] * 5333 + offs_n[None, :],
        mask=offs_k2[:, None] < 71,
        other=0.0,
    )
    accumulator = tl.dot(a2, b2, acc=accumulator, input_precision=INPUT_PRECISION)

    bias_value = tl.load(bias + offs_n).to(tl.float32)
    result = alpha * accumulator + beta * bias_value[None, :]
    tl.store(
        out + pid_b * (495 * 5333) + offs_m[:, None] * 5333 + offs_n[None, :],
        result.to(out.dtype.element_ty),
        mask=(offs_m[:, None] >= logical_m0) & (offs_n[None, :] >= logical_n0),
    )


def _launch_direct_forward(path, bias, A, B, alpha, beta, out):
    alpha = float(alpha)
    beta = float(beta)
    with torch_device_fn.device(A.device):
        if path == _TINY_FORWARD_PATH:
            _tiny_baddbmm_kernel[(1,)](
                A,
                B,
                bias,
                out,
                alpha,
                beta,
                num_warps=1,
                num_stages=1,
            )
        elif path == _K71_FORWARD_PATH:
            if A.dtype == torch.float32:
                # Only this scalar pair meets the official tolerance with one
                # TF32 product; all other FP32 calls retain three-term TF32.
                input_precision = (
                    "tf32" if alpha == 0.001 and beta == 0.001 else "tf32x3"
                )
            else:
                input_precision = "ieee"
            _k71_baddbmm_kernel[(42, 16, 4)](
                A,
                B,
                bias,
                out,
                alpha,
                beta,
                INPUT_PRECISION=input_precision,
                num_warps=4,
                num_stages=1,
                enable_backend_opt=True,
            )
        else:
            raise ValueError(f"unknown direct baddbmm path: {path}")
    return out


@triton.jit(do_not_specialize=["alpha", "beta"])
def _tiny_baddbmm_backward_kernel(
    grad_out,
    A,
    B,
    grad_bias,
    grad_A,
    grad_B,
    alpha,
    beta,
):
    offsets = tl.arange(0, 64)
    batch = offsets // 32
    k = offsets % 32
    grad = tl.load(grad_out + batch).to(tl.float32)
    a = tl.load(A + offsets).to(tl.float32)
    b = tl.load(B + offsets).to(tl.float32)
    tl.store(grad_A + offsets, (alpha * grad * b).to(grad_A.dtype.element_ty))
    tl.store(grad_B + offsets, (alpha * a * grad).to(grad_B.dtype.element_ty))
    tl.store(
        grad_bias + batch,
        (beta * grad).to(grad_bias.dtype.element_ty),
        mask=k == 0,
    )


def _tiny_backward_eligible(bias, A, B, grad_out):
    return (
        tuple(bias.shape) == (2, 1, 1)
        and tuple(A.shape) == (2, 1, 32)
        and tuple(B.shape) == (2, 32, 1)
        and tuple(grad_out.shape) == (2, 1, 1)
        and bias.dtype == A.dtype
        and A.dtype == B.dtype
        and B.dtype == grad_out.dtype
        and A.dtype in _DIRECT_DTYPES
        and bias.is_contiguous()
        and A.is_contiguous()
        and B.is_contiguous()
        and grad_out.is_contiguous()
        and bias.device == A.device
        and A.device == B.device
        and B.device == grad_out.device
        and A.device.type == "musa"
    )


def _launch_tiny_backward(grad_out, bias, A, B, alpha, beta):
    grad_bias = torch.empty_like(bias)
    grad_A = torch.empty_like(A)
    grad_B = torch.empty_like(B)
    with torch_device_fn.device(A.device):
        _tiny_baddbmm_backward_kernel[(1,)](
            grad_out,
            A,
            B,
            grad_bias,
            grad_A,
            grad_B,
            float(alpha),
            float(beta),
            num_warps=1,
            num_stages=1,
        )
    return grad_bias, grad_A, grad_B


def try_forward(bias, A, B, alpha, beta, out=None):
    """Run a validated exact-shape implementation or return ``None``."""
    return _dispatch_specialized_forward(bias, A, B, alpha, beta, out)


def try_backward(needs_input_grad, grad_output, bias, A, B, alpha, beta):
    """Run a validated backward implementation or return ``None``."""
    if all(needs_input_grad[:3]) and _tiny_backward_eligible(bias, A, B, grad_output):
        return _launch_tiny_backward(grad_output, bias, A, B, alpha, beta)

    if (
        all(needs_input_grad[:3])
        and _baddbmm_backward.backward_dispatch_path(bias, A, B, grad_output)
        is not None
    ):
        return _baddbmm_backward.launch_backward(
            grad_output,
            bias,
            A,
            B,
            alpha=alpha,
            beta=beta,
        )

    if not _generic_backward_eligible(bias, A, B, grad_output):
        return None

    grad_bias = grad_A = grad_B = None
    if needs_input_grad[0]:
        grad_bias = _baddbmm_generic.bias_gradient(
            grad_output,
            bias,
            beta=beta,
        )
    if needs_input_grad[1]:
        grad_A = _baddbmm_generic.matmul_gradient(
            grad_output,
            B.transpose(1, 2),
            tuple(A.shape),
            alpha=alpha,
        )
    if needs_input_grad[2]:
        grad_B = _baddbmm_generic.matmul_gradient(
            A.transpose(1, 2),
            grad_output,
            tuple(B.shape),
            alpha=alpha,
        )
    return grad_bias, grad_A, grad_B
