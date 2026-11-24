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

import logging
import math
import warnings

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.mm import get_higher_dtype
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def tensordot_mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
    # NOTE: accepted and unused. This kernel shares the "mm" tune configs
    # (see the libtuner decorator above), and on the backends whose own mm
    # kernel implements split-K those configs carry SPLIT_K. Declaring it
    # keeps the shared config list usable here; a kernel that reduces the
    # whole K extent in one pass IS the SPLIT_K == 1 case.
    SPLIT_K: tl.constexpr = 1,
):
    # Tiled matrix multiplication C = A @ B, where A is (M, K) and B is (K, N).
    # The contracted axes of tensordot are collapsed into K before launch.
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)

    if IS_FP64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        mask_k = rk < K
        a = tl.load(
            A + (ram[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn),
            mask=mask_k[:, None],
            other=0.0,
        )
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if IS_FP64:
            acc += tl.dot(a, b, allow_tf32=False)
        else:
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(C.dtype.element_ty)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


# ---------------------------------------------------------------------------
# gcu400 fast path. The upstream tensordot_mm_kernel above does all address
# arithmetic in int64 (rm/rn/rk widened via .to(tl.int64)), which the GCU400
# backend software-emulates: measured 22-370x slower than the native aten
# tensordot (bench kernel mode; e.g. 1024^3 fp16 22.8ms vs 18.8us, and the
# same 2D shape through flag_gems.mm differs 3-470x). tensordot_gemm_kernel
# below is the same tiled matmul with:
#   - int32-only index math (the host guards the int32 domain; the legacy
#     kernel above covers the overflow segment and fp64),
#   - strides taken as constexpr so each (layout, tile) pair compiles with
#     exact stride knowledge - strided operands such as the transposed views
#     used by the backward pass are consumed directly, no .contiguous()
#     materialization,
#   - EVEN_K / FULL_EVEN heuristic arms that strip the K-tail mask (and every
#     mask on tile-divisible shapes): a runtime branch around tl.dot degrades
#     to scalar code on GCU400, so the guard must be a constexpr,
#   - no SPLIT_K / tl.atomic_add arm: split-K reduction measured 30-240x
#     slower plus partial wrong values on GCU400 (perf_playbook veto).
# The shared gcu400 "mm" tune configs are all num_warps=4 while small/medium
# GEMMs on this part prefer num_warps=2 (router_gemm / linear precedent);
# both sets are kept in the libtuner space below.
# ---------------------------------------------------------------------------

_TENSORDOT_GEMM_CONFIGS = [
    # shared gcu400 "mm" configs (large-shape safety net, all num_warps=4)
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": 8},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    # num_warps=2 winners for small/medium GEMMs (blind spot of the shared
    # configs on this part; DSM <= 917504B for every entry)
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": 8},
        num_warps=2,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 512, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=1,
    ),
    # linear-session winners (gcu400 NT-GEMM w2 sweet spot extends to the
    # large shapes; fp32 medium-shape winner; small-MN/K-heavy candidates)
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 512, "GROUP_M": 16},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=2,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256, "GROUP_M": 8},
        num_warps=2,
        num_stages=2,
    ),
]

_INT32_ELEMENT_LIMIT = 2**31

# ---------------------------------------------------------------------------
# Native-kernel delegation (capture + call_boxed), same pattern as the local
# mm / bmm vendor ops. The aten::mm.out kernel is captured at import time -
# before any use_gems() registration replaces the PrivateUse1 entry - so
# call_boxed runs the original device kernel with zero dispatch overhead
# (mm/bmm precedent: verified 1.000-1.006x vs plain torch.mm). tensordot
# folds its operands down to a 2D (M, K) x (K, N) GEMM before launching, so
# the native GEMM kernel is a drop-in core for the segments where the Triton
# tile loses structurally (bench kernel mode, do_bench median w1000/r100,
# L600):
#   - 16-bit tiny-MN x long-K (M*N <= 2^16, work >= 2^22): the Triton launch
#     floor sits flat at ~9.0-9.7us (<=16 programs starve the 48 SPCs) while
#     the native library stays at its ~6.4-9.0us floor; split-K is vetoed
#     (30-240x worse plus partial wrong values, perf_playbook).
#   - bf16 mid/large (work >= 2^26): the native bf16 library runs ~15% ahead
#     of the Triton tile, which computes bf16 exactly like fp16.
#   - 16-bit / fp32 saturated (fp16 work >= 2^30, fp32 work >= 2^29): the
#     native primo++ pipeline is structurally ahead there (mm anchors:
#     1024^3 native wins, 384^3/512^3 Triton wins; the tensordot fp32 line
#     sits higher because (512, 512, 1024) = 2^28 still measures 1.44x).
#   - int32-overflow operands (any linear offset >= 2^31): the native kernel
#     addresses with int64, so delegation replaces the software-emulated
#     int64 legacy kernel for float16/32/bfloat16.
# The out= kernel always writes into an explicit result tensor - freshly
# allocated host-side for the functional path, the caller's storage for out=.
# That both skips the ~1.5-3us temp+copy the native tensordot.out composite
# pays and avoids the ~2us exposed pipeline stall of the boxed *allocating*
# mm.default on launch-bound shapes (measured: torch.empty + boxed mm.out
# lands on the native floor, the boxed allocating mm.default sits 2us above
# it). Non-contiguous out layouts keep the Triton constexpr-stride
# direct-write path (the native kernel rejects non-standard layouts,
# cuBLAS-style).
# ---------------------------------------------------------------------------
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_MM_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_MM_OUT = torch.library.get_kernel(
            torch.ops.aten.mm.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_MM_OUT = None

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)

# Dispatch lines: 16-bit tiny-MN arm, per-dtype saturation arms (see above).
_DELEGATE_TINY_MN_LIMIT = 1 << 16
_DELEGATE_TINY_MN_WORK = 1 << 22
_DELEGATE_BF16_WORK = 1 << 26
_DELEGATE_FP16_WORK = 1 << 30
_DELEGATE_FP32_WORK = 1 << 29
_DELEGATE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _should_delegate_to_native(a, b, M, N, K):
    if _NATIVE_MM_OUT is None:
        return False
    if a.device.type != runtime.device.name:
        return False
    if a.dtype != b.dtype or a.dtype not in _DELEGATE_DTYPES:
        return False
    work = M * N * K
    if a.dtype is torch.bfloat16:
        return work >= _DELEGATE_BF16_WORK or (
            M * N <= _DELEGATE_TINY_MN_LIMIT and work >= _DELEGATE_TINY_MN_WORK
        )
    if a.dtype is torch.float16:
        return work >= _DELEGATE_FP16_WORK or (
            M * N <= _DELEGATE_TINY_MN_LIMIT and work >= _DELEGATE_TINY_MN_WORK
        )
    # float32: only the saturated segment; Triton wins the whole mid range.
    return work >= _DELEGATE_FP32_WORK


def _tensordot_heur_even_k(args):
    return args["K"] % args["BLOCK_K"] == 0


def _tensordot_heur_full_even(args):
    return (
        args["K"] % args["BLOCK_K"] == 0
        and args["M"] % args["BLOCK_M"] == 0
        and args["N"] % args["BLOCK_N"] == 0
    )


@libentry()
@libtuner(
    configs=_TENSORDOT_GEMM_CONFIGS,
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(
    {
        "EVEN_K": _tensordot_heur_even_k,
        "FULL_EVEN": _tensordot_heur_full_even,
    }
)
@triton.jit
def tensordot_gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
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
    EVEN_K: tl.constexpr,
    FULL_EVEN: tl.constexpr,
):
    # Tiled matmul C = A @ B with A (M, K), B (K, N); int32 addressing only
    # (the host guarantees every linear offset stays below 2**31).
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if FULL_EVEN:
            a = tl.load(A)
            b = tl.load(B)
        elif EVEN_K:
            mask_a = (rm < M)[:, None]
            mask_b = (rn < N)[None, :]
            a = tl.load(A, mask_a, other=0)
            b = tl.load(B, mask_b, other=0)
        else:
            k_remaining = K - k * BLOCK_K
            mask_a = (rm < M)[:, None] & (rk < k_remaining)[None, :]
            mask_b = (rk < k_remaining)[:, None] & (rn < N)[None, :]
            a = tl.load(A, mask=mask_a, other=0)
            b = tl.load(B, mask=mask_b, other=0)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(C.dtype.element_ty)

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    if FULL_EVEN:
        tl.store(C, acc)
    else:
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C, acc, mask=mask)


def _matmul_2d(a, b, out=None):
    # Contract two 2D tensors (M, K) x (K, N) -> (M, N) with the Triton kernel.
    # Strided operands (e.g. the transposed views used by the backward pass)
    # are consumed directly through constexpr strides; only an operand with no
    # unit stride at all falls back to a contiguous copy.
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c_dtype = get_higher_dtype(a.dtype, b.dtype)

    if M == 0 or N == 0:
        # Empty result: nothing to compute, no kernel launch.
        if out is None:
            out = torch.empty((M, N), device=a.device, dtype=c_dtype)
        return out
    if K == 0:
        # Empty contraction: the result is all zeros (ATen semantics). Fill
        # host-side - a zero-iteration device loop clobbers the carried
        # accumulator on GCU400.
        if out is None:
            out = torch.empty((M, N), device=a.device, dtype=c_dtype)
        if tuple(out.shape) == (M, N):
            out.zero_()
        else:
            out.copy_(torch.zeros((M, N), device=a.device, dtype=c_dtype))
        return out

    if _should_delegate_to_native(a, b, M, N, K) and (
        out is None or out.dtype == a.dtype
    ):
        # Native GEMM core: bit-compatible with the aten production path. This
        # segment covers the shapes where every swept Triton tile loses to
        # the vendor library, and every int32-overflow operand (the native
        # kernel addresses in int64, replacing the emulated legacy kernel).
        # The out= kernel writes straight into the caller's (or the freshly
        # allocated) result tensor: measured to sit at the native-kernel
        # floor, while the boxed *allocating* mm.default pays ~2us of exposed
        # pipeline stall per call on launch-bound shapes (the safe-boxed
        # output allocation blocks on the in-flight kernel; a host-side
        # torch.empty hides inside the pipeline instead). Non-contiguous out
        # layouts fall through to the Triton constexpr-stride direct-write
        # kernel below (the captured native out= kernel cannot take them,
        # cuBLAS-style layout constraints).
        if out is None:
            out = torch.empty((M, N), device=a.device, dtype=c_dtype)
        if out.is_contiguous() and tuple(out.shape) == (M, N):
            # Direct write: also skips the temp+copy the native tensordot.out
            # composite pays on these shapes.
            _NATIVE_MM_OUT.call_boxed(_DEVICE_KEYSET, a, b, out=out)
            return out

    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=c_dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        if (
            a.dtype == torch.float64
            or M * K >= _INT32_ELEMENT_LIMIT
            or K * N >= _INT32_ELEMENT_LIMIT
            or M * N >= _INT32_ELEMENT_LIMIT
        ):
            # fp64 (on devices that enable it) or an operand whose linear
            # offsets exceed the int32 domain and could not be delegated:
            # stay on the upstream int64 kernel.
            tensordot_mm_kernel[grid](
                a,
                b,
                out,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                out.stride(0),
                out.stride(1),
                GROUP_M=8,
                IS_FP64=a.dtype == torch.float64,
            )
        else:
            tensordot_gemm_kernel[grid](
                a,
                b,
                out,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                out.stride(0),
                out.stride(1),
            )
    return out


def _normalize_dims(dims, ndim):
    # Map (possibly negative) dim indices into the range [0, ndim).
    # Raise error for out-of-range dimensions.
    normalized = []
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})"
            )
        normalized.append(d if d >= 0 else d + ndim)
    return normalized


def _invert_permutation(perm):
    # Invert a permutation so a permuted tensor can be mapped back to its
    # original axis order. perm[i] is the source axis for destination axis i.
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _resolve_contraction(self, other, dims_self, dims_other):
    # Normalize contracted dims and reconcile their sizes following the ATen
    # tensordot contract: a contracted axis may be size 1 on either side, in
    # which case it is broadcast (and later reduced) rather than rejected. Only
    # unequal, non-1 sizes are an error. Returns normalized dim lists together
    # with the (possibly expanded) tensors that share matching contracted sizes.
    dims_self = _normalize_dims(list(dims_self), self.ndim)
    dims_other = _normalize_dims(list(dims_other), other.ndim)

    if len(dims_self) != len(dims_other):
        raise RuntimeError(
            "tensordot: number of contracted dims must match, but got "
            f"{len(dims_self)} and {len(dims_other)}"
        )

    self_shape = list(self.shape)
    other_shape = list(other.shape)
    for ds, do in zip(dims_self, dims_other):
        s1 = self.shape[ds]
        s2 = other.shape[do]
        if s1 == s2:
            continue
        if s1 == 1:
            self_shape[ds] = s2
        elif s2 == 1:
            other_shape[do] = s1
        else:
            raise RuntimeError(
                "tensordot: contracted dimensions need to match, but got "
                f"size {s1} in dim {ds} of self and size {s2} in dim {do} of "
                "other"
            )

    self_e = self.expand(self_shape) if self_shape != list(self.shape) else self
    other_e = other.expand(other_shape) if other_shape != list(other.shape) else other
    return dims_self, dims_other, self_e, other_e


def _copy_tensordot_out(result: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Validate, resize, and copy a computed tensordot result into an out tensor."""
    if out.device != result.device:
        raise RuntimeError(
            "tensordot: Expected the output tensor to be on the same device as "
            f"the result ({result.device}), but got {out.device}"
        )
    if out.dtype != result.dtype:
        raise RuntimeError(
            f"tensordot: Expected the output tensor to have dtype {result.dtype}, "
            f"but got {out.dtype}"
        )
    if tuple(out.shape) != tuple(result.shape):
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(result.shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=3,
            )
        out.resize_(result.shape)
    out.copy_(result)
    return out


def _tensordot_impl(self, other, dims_self, dims_other, out=None):
    # Validate supported dtypes - only floating point types
    supported_dtypes = {torch.float16, torch.float32, torch.bfloat16}
    if runtime.device.support_fp64:
        supported_dtypes.add(torch.float64)

    if self.dtype not in supported_dtypes:
        raise RuntimeError(
            f"tensordot: unsupported dtype {self.dtype}. "
            f"Only floating point dtypes are supported."
        )

    if other.dtype not in supported_dtypes:
        raise RuntimeError(
            f"tensordot: unsupported dtype {other.dtype}. "
            f"Only floating point dtypes are supported."
        )

    # Validate dtype and device match
    if self.dtype != other.dtype:
        raise RuntimeError(
            f"tensordot: expected self and other to have the same dtype, "
            f"but got {self.dtype} and {other.dtype}"
        )
    if self.device != other.device:
        raise RuntimeError(
            f"tensordot: expected self and other to be on the same device, "
            f"but got {self.device} and {other.device}"
        )

    dims_self, dims_other, self, other = _resolve_contraction(
        self, other, dims_self, dims_other
    )

    # Free (non-contracted) dims, preserving their original order.
    free_self = [d for d in range(self.ndim) if d not in dims_self]
    free_other = [d for d in range(other.ndim) if d not in dims_other]

    free_self_sizes = [self.shape[d] for d in free_self]
    free_other_sizes = [other.shape[d] for d in free_other]

    M = math.prod(free_self_sizes) if free_self_sizes else 1
    N = math.prod(free_other_sizes) if free_other_sizes else 1
    K = math.prod([self.shape[d] for d in dims_self]) if dims_self else 1

    # Reorder so contracted dims are collapsed into a single axis:
    #   self  -> (free_self..., contracted...)  reshaped to (M, K)
    #   other -> (contracted..., free_other...) reshaped to (K, N)
    a2d = self.permute(free_self + dims_self).contiguous().reshape(M, K)
    b2d = other.permute(dims_other + free_other).contiguous().reshape(K, N)

    result_shape = free_self_sizes + free_other_sizes

    if out is not None:
        # Validate + resize out first (aten out= contract: raise on dtype or
        # device mismatch, resize-with-warning on shape mismatch), then write
        # the GEMM straight into out's storage through a 2D view when its
        # layout allows one - no temporary result, no epilogue copy. Exotic
        # non-collapsible out layouts keep the resize_ + copy_ path.
        if out.device != self.device:
            raise RuntimeError(
                "tensordot: Expected the output tensor to be on the same device as "
                f"the result ({self.device}), but got {out.device}"
            )
        if out.dtype != a2d.dtype:
            raise RuntimeError(
                f"tensordot: Expected the output tensor to have dtype {a2d.dtype}, "
                f"but got {out.dtype}"
            )
        if tuple(out.shape) != tuple(result_shape):
            if out.numel() != 0:
                warnings.warn(
                    "An output with one or more elements was resized since it had "
                    f"shape {list(out.shape)}, which does not match the required "
                    f"output shape {list(result_shape)}. This behavior is deprecated, "
                    "and in a future PyTorch release outputs will not be resized "
                    "unless they have zero elements. You can explicitly reuse an out "
                    "tensor t by resizing it, inplace, to zero elements with "
                    "t.resize_(0).",
                    UserWarning,
                    stacklevel=3,
                )
            out.resize_(result_shape)
        if out.dim() == 2 and tuple(out.shape) == (M, N):
            out_2d = out
        elif out.is_contiguous():
            out_2d = out.view(M, N)
        else:
            out_2d = None
        if out_2d is not None:
            _matmul_2d(a2d, b2d, out=out_2d)
            return out
        res = _matmul_2d(a2d, b2d).reshape(result_shape)
        return _copy_tensordot_out(res, out)

    res = _matmul_2d(a2d, b2d)
    return res.reshape(result_shape)


class TensordotFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, other, dims_self, dims_other):
        logger.debug("GEMS_ENFLAME TENSORDOT_FORWARD")

        # Save the raw inputs and their original contracted-dim sizes so the
        # backward pass can broadcast for the matmul and then reduce the
        # gradient back to any size-1 contracted axis, matching ATen.
        dims_self_n, dims_other_n, _, _ = _resolve_contraction(
            self, other, dims_self, dims_other
        )
        ctx.dims_self = dims_self_n
        ctx.dims_other = dims_other_n
        ctx.save_for_backward(self, other)

        return _tensordot_impl(self, other, dims_self, dims_other)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_ENFLAME TENSORDOT_BACKWARD")

        self, other = ctx.saved_tensors
        dims_self = ctx.dims_self
        dims_other = ctx.dims_other

        # Original (pre-broadcast) contracted sizes, used to reduce the grad
        # back to a size-1 axis after the matmul over the broadcast size.
        self_orig_sizes = {d: self.shape[d] for d in dims_self}
        other_orig_sizes = {d: other.shape[d] for d in dims_other}

        # Broadcast contracted dims so both operands share matching sizes.
        _, _, self_e, other_e = _resolve_contraction(self, other, dims_self, dims_other)

        grad_self = None
        grad_other = None

        free_self = [d for d in range(self_e.ndim) if d not in dims_self]
        free_other = [d for d in range(other_e.ndim) if d not in dims_other]

        free_self_sizes = [self_e.shape[d] for d in free_self]
        free_other_sizes = [other_e.shape[d] for d in free_other]
        M = math.prod(free_self_sizes) if free_self_sizes else 1
        N = math.prod(free_other_sizes) if free_other_sizes else 1
        K = math.prod([self_e.shape[d] for d in dims_self]) if dims_self else 1

        grad_2d = grad_output.reshape(M, N)

        if ctx.needs_input_grad[0]:
            b2d = other_e.permute(dims_other + free_other).contiguous().reshape(K, N)
            grad_a2d = _matmul_2d(grad_2d, b2d.transpose(0, 1))
            grad_self = grad_a2d.reshape(self_e.permute(free_self + dims_self).shape)
            grad_self = grad_self.permute(
                _invert_permutation(free_self + dims_self)
            ).contiguous()
            # Reduce over any contracted axis that was size 1 on self.
            for d in dims_self:
                if self_orig_sizes[d] == 1 and self_e.shape[d] != 1:
                    grad_self = grad_self.sum(d, keepdim=True)

        if ctx.needs_input_grad[1]:
            a2d = self_e.permute(free_self + dims_self).contiguous().reshape(M, K)
            grad_b2d = _matmul_2d(a2d.transpose(0, 1), grad_2d)
            grad_other = grad_b2d.reshape(
                other_e.permute(dims_other + free_other).shape
            )
            grad_other = grad_other.permute(
                _invert_permutation(dims_other + free_other)
            ).contiguous()
            # Reduce over any contracted axis that was size 1 on other.
            for d in dims_other:
                if other_orig_sizes[d] == 1 and other_e.shape[d] != 1:
                    grad_other = grad_other.sum(d, keepdim=True)

        return grad_self, grad_other, None, None


def tensordot(self, other, dims_self, dims_other):
    logger.debug("GEMS TENSORDOT")
    # The autograd Function only serves direct python calls with grad-requiring
    # operands (a dispatched call with grad takes the composite decomposition
    # instead and never reaches this kernel). Every other caller - inference,
    # no-grad contexts, non-grad operands - skips the Function.apply
    # machinery: its host-side cost is ~2us of exposed launch latency on the
    # launch-bound shapes (measured: same GEMM through the out= path, which
    # has no wrapper, sits at the native kernel floor).
    if torch.is_grad_enabled() and (self.requires_grad or other.requires_grad):
        return TensordotFunction.apply(self, other, dims_self, dims_other)
    return _tensordot_impl(self, other, dims_self, dims_other)


def tensordot_out(self, other, dims_self, dims_other, *, out):
    logger.debug("GEMS_ENFLAME TENSORDOT_OUT")
    _tensordot_impl(self, other, dims_self, dims_other, out=out)
    return out
