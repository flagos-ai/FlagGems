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
import warnings

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

# gcu400 dispatch policy (measured on L600, bench kernel mode do_bench median
# w1000/r100, job j20261110175400_f6fd2577 + j20261116150858_246bf2f5):
#  - fp32 chains of 2-3 tensors route each multiplication through the lean
#    small-GEMM kernel (deterministic wave-aware host config, see the lean
#    block below) when every mm is a small GEMM, falling back to the tuned
#    vendor mm kernel otherwise. fp16 runs lean for single mms only (2-tensor
#    chains): the tuned kernel beat native aten::mm per-mm on fp16
#    (0.63-0.74x) and fp32 (0.75-0.94x); the lean kernel removes the
#    remaining ~3-5us/launch of libtuner+wrapper Python tax (core 2-3 tensor
#    chains measured 1.3-2.3x vs native). fp16 3-tensor chains keep the
#    tuned path so the fp16-rounded intermediate stays bit-stable within the
#    reference tolerance (lean accumulation order measured up to 0.25 abs
#    over tolerance on [128,512,64,256]).
#  - fp32 4-tensor chains with all dims <= 512 take the tight _four_dot_fast
#    path (closed-form chain order + 3 lean launches, 1.8-2.2x vs native);
#    anything the gate rejects keeps the policy below.
#  - bf16 (native GEMM wins per-mm, lean only ties), fp64/exotic dtypes and
#    chains of >= 5 tensors delegate to the native aten::linalg_multi_dot
#    captured at import time - the same pattern as the public
#    _linalg_eigvals / linalg_matrix_sqrth and the local linalg_cond. The
#    private 32x32x32 Triton chain stays as the capture-unavailable /
#    CPU-redispatch fallback.
from .mm import mm as _tuned_mm
from .mm import mm_out as _tuned_mm_out

# ---------------------------------------------------------------------------
# Lean small-GEMM fast path (measured on L600, job j20261116150858_246bf2f5).
# Core multi_dot chains are tiny GEMMs (dims <= 512); the shared tuned-mm
# path pays several us of libtuner/heuristics/wrapper Python per mm, which
# dominates these ~5-25us calls. The lean path keeps mm.py's numerics (fp32
# accumulator, allow_tf32=False, dtype-rounded intermediates == the aten
# chain semantics) and picks a deterministic config from host metadata only:
#   - tiles: smallest of (32,32),(64,32),(32,64),(64,64),(128,64),(64,128)
#     whose grid fits one 48-CTA wave; a (32,32) grid above 48 CTAs runs two
#     waves and measured ~1.6x slower ((512,64)x(64,128) fp16 8.0 -> 4.9us
#     after the wave-aware pick).
#   - BLOCK_K: 128/64/32/16 by divisibility; the K loop is a static_range
#     unroll (gcu400 miscompiles loop-carried fp32 tl.dot accumulators in
#     dynamic range loops - see mm_heur_k_blocks).
#   - num_warps=2 / num_stages=1 beat the shared nw4 configs on every probed
#     small shape (fp32 (64,256)x(256,128) 11.7 -> 5.8us end to end).
#   - FULL_EVEN drops every load/store mask when M/N/K are tile-divisible.
# 2-3 tensor chains keep the regular wrapper (validate overlaps device work
# there); fp32 4-tensor chains need a tight host path because the generic
# validate + DP + recursion adds ~16-19us of in-context host time (measured
# 37-48us vs 17-21us hand-rolled), so _four_dot_fast runs a zero-allocation
# closed-form chain order with the cheap gate doubling as the happy-path
# validation (anything the gate rejects falls back to the generic paths,
# which raise the canonical errors or delegate). fp16 chains of 3+ tensors
# and all bf16/fp64 chains stay on the tuned/delegated paths: a 1-ulp
# difference on an fp16-rounded intermediate is amplified by the next GEMM
# beyond the reference tolerance (0.25-4.0 abs measured), so only single-mm
# fp16 calls and fp32 chains (no rounding amplification, ~1e-5) run lean.
# ---------------------------------------------------------------------------

_LEAN_DTYPES = (torch.float16, torch.float32)
_LEAN_MAX_DIM = 512
_LEAN_WAVE = 48


def _lean_block_k(K):
    if K % 128 == 0:
        return 128
    if K % 64 == 0:
        return 64
    if K % 32 == 0:
        return 32
    if K % 16 == 0:
        return 16
    return min(128, max(16, triton.next_power_of_2(K)))


def _lean_tiles(M, N):
    if (M + 31) // 32 * ((N + 31) // 32) <= _LEAN_WAVE:
        return 32, 32
    if (M + 63) // 64 * ((N + 31) // 32) <= _LEAN_WAVE:
        return 64, 32
    if (M + 31) // 32 * ((N + 63) // 64) <= _LEAN_WAVE:
        return 32, 64
    if (M + 63) // 64 * ((N + 63) // 64) <= _LEAN_WAVE:
        return 64, 64
    if (M + 127) // 128 * ((N + 63) // 64) <= _LEAN_WAVE:
        return 128, 64
    return 64, 128


@triton.jit
def _lean_mm_kernel(
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
    K_BLOCKS: tl.constexpr,
    EVEN_K: tl.constexpr,
    FULL_EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.static_range(K_BLOCKS):
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


def _lean_mm(left, right, out=None):
    """Launch the lean kernel; caller has already validated eligibility."""
    M, K = left.shape
    N = right.shape[1]
    if left.stride(0) > 1 and left.stride(1) > 1:
        left = left.contiguous()
    if right.stride(0) > 1 and right.stride(1) > 1:
        right = right.contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=left.dtype, device=left.device)
    block_m, block_n = _lean_tiles(M, N)
    block_k = _lean_block_k(K)
    grid = ((M + block_m - 1) // block_m * ((N + block_n - 1) // block_n),)
    k_blocks = (K + block_k - 1) // block_k
    _lean_mm_kernel[grid](
        left,
        right,
        out,
        M,
        N,
        K,
        left.stride(0),
        left.stride(1),
        right.stride(0),
        right.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        K_BLOCKS=k_blocks,
        EVEN_K=(K % block_k == 0),
        FULL_EVEN=(M % block_m == 0 and N % block_n == 0 and K % block_k == 0),
        num_warps=2,
        num_stages=1,
    )
    return out


def _lean_mm_ok(left, right, out):
    if left.dtype not in _LEAN_DTYPES:
        return False
    M, K = left.shape
    N = right.shape[1]
    if M > _LEAN_MAX_DIM or N > _LEAN_MAX_DIM or K > _LEAN_MAX_DIM:
        return False
    if M < 1 or N < 1 or K < 1:
        return False
    if right.dtype is not left.dtype:
        return False
    if out is not None:
        if out.dtype is not left.dtype or out.device != left.device:
            return False
        if out.shape[0] != M or out.shape[1] != N:
            return False
    return True


def _four_dot_gate(tensors):
    """Metadata gate for the tight 4-tensor fp16/fp32 path.

    Returns the (d0..d4) dimension chain or None when the generic path
    must handle the call (non-2D tensors, heterogeneous dtype/device,
    incompatible chain, oversized dims, CPU tensors).
    """
    if len(tensors) != 4:
        return None
    t0, t1, t2, t3 = tensors
    dtype = t0.dtype
    # fp32 only: fp16 4-chains stay delegated - any non-bit-identical fp16
    # chain deviates from the native reference beyond tolerance once the
    # fp16-rounded intermediates are amplified (measured max 4.0 abs).
    if dtype is not torch.float32:
        return None
    if t0.dim() != 2 or t1.dim() != 2 or t2.dim() != 2 or t3.dim() != 2:
        return None
    if t1.dtype is not dtype or t2.dtype is not dtype or t3.dtype is not dtype:
        return None
    device = t0.device
    if t1.device != device or t2.device != device or t3.device != device:
        return None
    if device.type != runtime.device.name:
        return None
    d0, d1 = t0.shape
    d1b, d2 = t1.shape
    d2b, d3 = t2.shape
    d3b, d4 = t3.shape
    if d1 != d1b or d2 != d2b or d3 != d3b:
        return None
    if (
        d0 < 1
        or d1 < 1
        or d2 < 1
        or d3 < 1
        or d4 < 1
        or d0 > _LEAN_MAX_DIM
        or d1 > _LEAN_MAX_DIM
        or d2 > _LEAN_MAX_DIM
        or d3 > _LEAN_MAX_DIM
        or d4 > _LEAN_MAX_DIM
    ):
        return None
    return d0, d1, d2, d3, d4


def _four_dot_fast(tensors, out=None):
    """Tight 4-tensor chain: closed-form optimal order + lean launches.

    Returns the result tensor (``out`` when given) or None when the gate
    rejects the call - the caller then falls back to the generic paths.
    """
    dims = _four_dot_gate(tensors)
    if dims is None:
        return None
    d0, d1, d2, d3, d4 = dims
    t0, t1, t2, t3 = tensors
    if out is not None:
        if out.dtype is not t0.dtype or out.device != t0.device:
            return None
        if out.dim() != 2 or out.shape[0] != d0 or out.shape[1] != d4:
            return None
    # closed-form optimal parenthesization of the 5 possible chain orders
    c_left_left = d0 * d1 * d2 + d0 * d2 * d3 + d0 * d3 * d4  # ((t0 t1) t2) t3
    c_mid_left = d1 * d2 * d3 + d0 * d1 * d3 + d0 * d3 * d4  # (t0 (t1 t2)) t3
    c_split = d0 * d1 * d2 + d2 * d3 * d4 + d0 * d2 * d4  # (t0 t1) (t2 t3)
    c_mid_right = d1 * d2 * d3 + d1 * d3 * d4 + d0 * d1 * d4  # t0 ((t1 t2) t3)
    c_right_right = d2 * d3 * d4 + d1 * d2 * d4 + d0 * d1 * d4  # t0 (t1 (t2 t3))
    best = 0
    cost = c_left_left
    if c_mid_left < cost:
        best, cost = 1, c_mid_left
    if c_split < cost:
        best, cost = 2, c_split
    if c_mid_right < cost:
        best, cost = 3, c_mid_right
    if c_right_right < cost:
        best = 4
    if best == 0:
        x = _lean_mm(t0, t1)
        x = _lean_mm(x, t2)
        return _lean_mm(x, t3, out)
    if best == 1:
        x = _lean_mm(t1, t2)
        x = _lean_mm(t0, x)
        return _lean_mm(x, t3, out)
    if best == 2:
        x = _lean_mm(t0, t1)
        y = _lean_mm(t2, t3)
        return _lean_mm(x, y, out)
    if best == 3:
        x = _lean_mm(t1, t2)
        x = _lean_mm(x, t3)
        return _lean_mm(t0, x, out)
    x = _lean_mm(t2, t3)
    x = _lean_mm(t1, x)
    return _lean_mm(t0, x, out)


logger = logging.getLogger(__name__)

_CPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_MULTI_DOT = None
_NATIVE_MULTI_DOT_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_MULTI_DOT = torch.library.get_kernel(
            torch.ops.aten.linalg_multi_dot.default, runtime.device.dispatch_key
        )
        _NATIVE_MULTI_DOT_OUT = torch.library.get_kernel(
            torch.ops.aten.linalg_multi_dot.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_MULTI_DOT = None
        _NATIVE_MULTI_DOT_OUT = None


_TRITON_CHAIN_DTYPES = (torch.float16, torch.float32)
_TRITON_CHAIN_MAX_TENSORS = 3


def _should_delegate_to_native(tensors, native_kernel):
    """Metadata-only dispatch: pick the native composite where it is faster."""
    if native_kernel is None:
        return False
    if not tensors:
        return False
    first = tensors[0]
    if first.device.type == "cpu":
        return False
    if len(tensors) > _TRITON_CHAIN_MAX_TENSORS:
        return True
    return first.dtype not in _TRITON_CHAIN_DTYPES


@libentry()
@triton.jit
def _multi_dot_mm_kernel(
    left,
    right,
    output,
    rows,
    columns,
    inner,
    stride_lm,
    stride_lk,
    stride_rk,
    stride_rn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_n = tl.cdiv(columns, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N), dtype=tl.float64 if IS_FP64 else tl.float32
    )
    for start_k in range(0, tl.cdiv(inner, BLOCK_K)):
        current_k = start_k * BLOCK_K + offsets_k
        left_values = tl.load(
            left + offsets_m[:, None] * stride_lm + current_k[None, :] * stride_lk,
            mask=(offsets_m[:, None] < rows) & (current_k[None, :] < inner),
            other=0.0,
        )
        right_values = tl.load(
            right + current_k[:, None] * stride_rk + offsets_n[None, :] * stride_rn,
            mask=(current_k[:, None] < inner) & (offsets_n[None, :] < columns),
            other=0.0,
        )
        accumulator += tl.dot(left_values, right_values, allow_tf32=False)

    tl.store(
        output + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on,
        accumulator,
        mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < columns),
    )


def _matrix_multiply(left, right, out=None, allow_lean=True):
    """Multiply without TF32 so results match ATen's multi_dot precision.

    ``allow_lean`` keeps the lean kernel out of multi-mm fp16 chains: a
    1-ulp difference on an fp16-rounded intermediate is amplified by the
    following GEMMs and can exceed the reference tolerance (measured up to
    0.25 abs on [128,512,64,256] and 4.0 on [256,128,512,64,256] vs the
    native chain), while fp32 intermediates carry no such amplification
    (~1e-5 end to end). Single-mm calls may always use lean.
    """
    if left.device.type == "cpu":
        if out is not None:
            return torch.ops.aten.mm.out.redispatch(_CPU_KEYSET, left, right, out=out)
        return torch.ops.aten.mm.default.redispatch(_CPU_KEYSET, left, right)

    if left.dtype in (torch.float16, torch.bfloat16, torch.float32):
        if allow_lean and _lean_mm_ok(left, right, out):
            return _lean_mm(left, right, out)
        if out is None:
            return _tuned_mm(left, right)
        return _tuned_mm_out(left, right, out=out)

    rows, inner = left.shape
    columns = right.shape[1]
    if out is None:
        out = torch.empty((rows, columns), dtype=left.dtype, device=left.device)
    grid = (triton.cdiv(rows, 32) * triton.cdiv(columns, 32),)
    with torch_device_fn.device(left.device):
        _multi_dot_mm_kernel[grid](
            left,
            right,
            out,
            rows,
            columns,
            inner,
            left.stride(0),
            left.stride(1),
            right.stride(0),
            right.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=32,
            IS_FP64=left.dtype == torch.float64,
            num_warps=4,
        )
    return out


def _validate_and_prepare(tensors):
    num_tensors = len(tensors)
    if num_tensors < 2:
        raise RuntimeError(
            f"multi_dot(): expected at least 2 tensors but got {num_tensors}"
        )

    arrays = list(tensors)
    first = arrays[0]
    last = arrays[-1]
    if first.ndim not in (1, 2):
        raise RuntimeError(
            f"multi_dot(): the first tensor must be 1D or 2D but got {first.ndim}D"
        )
    if last.ndim not in (1, 2):
        raise RuntimeError(
            f"multi_dot(): the last tensor must be 1D or 2D but got {last.ndim}D"
        )

    for index, tensor in enumerate(arrays[1:-1], 1):
        if tensor.ndim != 2:
            raise RuntimeError(
                f"multi_dot(): tensor {index} must be 2D but got {tensor.ndim}D"
            )

    for index, tensor in enumerate(arrays[1:], 1):
        if tensor.dtype != first.dtype:
            raise RuntimeError(
                "multi_dot(): all tensors must have be the same dtype but "
                f"tensor 0 is {first.dtype} and tensor {index} {tensor.dtype}"
            )
        if tensor.device != first.device:
            raise RuntimeError(
                "multi_dot(): all tensors must be on the same device but "
                f"tensor 0 is on {first.device} and tensor {index} is on {tensor.device}"
            )

    first_was_1d = first.ndim == 1
    last_was_1d = last.ndim == 1
    if first_was_1d:
        arrays[0] = first.unsqueeze(0)
    if last_was_1d:
        arrays[-1] = last.unsqueeze(1)

    for index in range(num_tensors - 1):
        if arrays[index].shape[1] != arrays[index + 1].shape[0]:
            raise RuntimeError(
                f"multi_dot(): tensors {index} and {index + 1} with shapes "
                f"{list(tensors[index].shape)} and {list(tensors[index + 1].shape)} "
                "cannot be multiplied"
            )

    output_shape = [arrays[0].shape[0], arrays[-1].shape[1]]
    if first_was_1d:
        output_shape.pop(0)
    if last_was_1d:
        output_shape.pop(-1)
    return arrays, tuple(output_shape)


def _matrix_chain_order(arrays):
    num_tensors = len(arrays)
    dimensions = [arrays[0].shape[0]] + [array.shape[1] for array in arrays]
    costs = [[0] * num_tensors for _ in range(num_tensors)]
    splits = [[0] * num_tensors for _ in range(num_tensors)]

    for chain_length in range(2, num_tensors + 1):
        for start in range(num_tensors - chain_length + 1):
            end = start + chain_length - 1
            best_cost = None
            for split in range(start, end):
                cost = (
                    costs[start][split]
                    + costs[split + 1][end]
                    + dimensions[start] * dimensions[split + 1] * dimensions[end + 1]
                )
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    splits[start][end] = split
            costs[start][end] = best_cost
    return splits


def _multiply_chain(arrays, splits, start, end, out=None, allow_lean=True):
    if start == end:
        return arrays[start]

    split = splits[start][end]
    left = _multiply_chain(arrays, splits, start, split, allow_lean=allow_lean)
    right = _multiply_chain(arrays, splits, split + 1, end, allow_lean=allow_lean)
    if out is not None:
        return _matrix_multiply(left, right, out=out, allow_lean=allow_lean)
    return _matrix_multiply(left, right, allow_lean=allow_lean)


def _multiply_three(arrays, out=None, allow_lean=True):
    a, b, c = arrays
    rows, inner_ab = a.shape
    inner_bc, columns = c.shape
    left_cost = rows * inner_bc * (inner_ab + columns)
    right_cost = inner_ab * columns * (rows + inner_bc)

    if left_cost > right_cost:
        right = _matrix_multiply(b, c, allow_lean=allow_lean)
        return _matrix_multiply(a, right, out=out, allow_lean=allow_lean)

    left = _matrix_multiply(a, b, allow_lean=allow_lean)
    return _matrix_multiply(left, c, out=out, allow_lean=allow_lean)


def _multi_dot_impl(arrays, out=None):
    num_tensors = len(arrays)
    if num_tensors == 2:
        return _matrix_multiply(arrays[0], arrays[1], out=out)
    allow_lean = arrays[0].dtype is torch.float32
    if num_tensors == 3:
        return _multiply_three(arrays, out=out, allow_lean=allow_lean)

    splits = _matrix_chain_order(arrays)
    return _multiply_chain(
        arrays, splits, 0, num_tensors - 1, out=out, allow_lean=allow_lean
    )


def linalg_multi_dot(tensors):
    logger.debug("GEMS_ENFLAME LINALG_MULTI_DOT")
    if len(tensors) == 4:
        fast = _four_dot_fast(tensors)
        if fast is not None:
            return fast
    if _should_delegate_to_native(tensors, _NATIVE_MULTI_DOT):
        return _NATIVE_MULTI_DOT.call_boxed(_DEVICE_KEYSET, list(tensors))
    arrays, output_shape = _validate_and_prepare(tensors)
    result = _multi_dot_impl(arrays)
    return result.view(output_shape)


def linalg_multi_dot_out(tensors, *, out):
    logger.debug("GEMS_ENFLAME LINALG_MULTI_DOT_OUT")
    if len(tensors) == 4:
        fast = _four_dot_fast(tensors, out)
        if fast is not None:
            return fast
    if _should_delegate_to_native(tensors, _NATIVE_MULTI_DOT_OUT):
        return _NATIVE_MULTI_DOT_OUT.call_boxed(_DEVICE_KEYSET, list(tensors), out=out)
    arrays, output_shape = _validate_and_prepare(tensors)
    first = arrays[0]
    if out.dtype != first.dtype:
        raise RuntimeError(
            f"multi_dot(): expected out tensor to have dtype {first.dtype} "
            f"but got {out.dtype}"
        )
    if out.device != first.device:
        raise RuntimeError(
            f"multi_dot(): expected out tensor to be on device {first.device} "
            f"but got {out.device}"
        )

    if tuple(out.shape) != output_shape:
        if out.numel() != 0:
            warnings.warn(
                "An output with one or more elements was resized since it had "
                f"shape {list(out.shape)}, which does not match the required "
                f"output shape {list(output_shape)}. This behavior is deprecated, "
                "and in a future PyTorch release outputs will not be resized "
                "unless they have zero elements. You can explicitly reuse an out "
                "tensor t by resizing it, inplace, to zero elements with "
                "t.resize_(0).",
                UserWarning,
                stacklevel=2,
            )
        out.resize_(output_shape)

    matrix_out = out.view(arrays[0].shape[0], arrays[-1].shape[1])
    _multi_dot_impl(arrays, out=matrix_out)
    return out
