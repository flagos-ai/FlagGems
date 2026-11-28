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

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native-kernel delegation (capture + call_boxed), same pattern as the local
# bmm / mm / linalg_matrix_power / linalg_multi_dot vendor ops. The kernels
# are captured at import time - before any use_gems() registration replaces
# the PrivateUse1 entry - so call_boxed runs the original device kernel
# bit-identically with ~zero dispatch overhead.
#
# Chain-mode anchors (matrix_power n=5 = 3 GEMMs, batch=1, do_bench median
# w1000/r100), native mm chain vs the vendored Triton chain below:
#   bf16:  native device chain wins at every probed size - m=64: 12.5 vs
#          34.3us, m=384: 17.6 vs 37.6us, m=2048: 198 vs 270us.
#   fp16:  Triton keeps the small end (m=384: 35.7 vs 38.1us), native wins
#          from ~645^3 up (m=2048: 197 vs 271us = 1.38x).
#   fp32:  Triton keeps the small end (m=384: 55.3 vs 36.3us = 1.52x),
#          native wins from ~645^3 up (m=1024: 117.5 vs 148.7us = 1.27x).
#
# CHANNEL CAVEAT (decisive for the dispatch lines below): when this op is
# entered through the aten dispatcher (the production / bench channel, i.e.
# a Python kernel invoked from C++), every in-host call_boxed pays ~12us of
# dispatch re-entry overhead that a plain-Python caller does not (direct
# call: 17.5us for the whole bf16-384 chain; dispatcher channel: 53.9us;
# Triton launches are exempt - libentry does not re-enter the dispatcher).
# The overhead hides once per-GEMM device time exceeds it, so delegation is
# restricted to saturated sizes: half dtypes m >= 1024 (18.9us/GEMM, 57%
# headroom), fp32 m >= 768 (26us/GEMM). Below the line the Triton arm wins
# the dispatcher channel (bf16-384: Triton 44.3 vs delegated 53.9us).
# ---------------------------------------------------------------------------
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_MM = None
_NATIVE_MM_OUT = None
_NATIVE_BMM = None
_NATIVE_BMM_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_MM = torch.library.get_kernel(
            torch.ops.aten.mm.default, runtime.device.dispatch_key
        )
        _NATIVE_MM_OUT = torch.library.get_kernel(
            torch.ops.aten.mm.out, runtime.device.dispatch_key
        )
        _NATIVE_BMM = torch.library.get_kernel(
            torch.ops.aten.bmm.default, runtime.device.dispatch_key
        )
        _NATIVE_BMM_OUT = torch.library.get_kernel(
            torch.ops.aten.bmm.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_MM = None
        _NATIVE_MM_OUT = None
        _NATIVE_BMM = None
        _NATIVE_BMM_OUT = None

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)

# Saturation lines for the square chain (m x m products, batch=1): the
# per-GEMM device time must exceed the ~12us dispatcher-channel re-entry
# cost of call_boxed. Measured anchors on both sides of each line:
#   half dtypes: m=768 native 44.8 vs Triton 46.0us (wash), m=1024 56.5 vs
#                59.0us (native), m=2048 198 vs 270us (native).
#   fp32:        m=512 Triton 49.3 vs native 64.1us, m=768 native 78.3 vs
#                Triton 87.1us, m=1024 native 117.5 vs 148.7us.
_NATIVE_SAT_M_HALF = 1024
_NATIVE_SAT_M_FP32 = 768


def _should_delegate_to_native(A, B):
    """True when this chain product should run on the captured native GEMM."""
    if _NATIVE_MM is None or _NATIVE_BMM is None:
        return False
    if A.device.type != runtime.device.name:
        return False
    if A.dtype == torch.float64:
        # gcu400 has no fp64 datapath; keep the vendored fp64 tile config.
        return False
    Batch, M, K = A.shape
    N = B.shape[2]
    if Batch == 1:
        # 2D products go through native mm - full precision across the whole
        # domain. (Native bmm picks a TF32-approximate path below 512, see the
        # linalg_matrix_power vendor op; well-conditioned inputs show it.)
        if A.dtype == torch.float32:
            return M >= _NATIVE_SAT_M_FP32
        return M >= _NATIVE_SAT_M_HALF
    # Batched chains mirror the backend bmm() hybrid line (work >= 2^29) and
    # additionally require every matrix dim >= 512 so the native bmm TF32
    # path is never selected; per-GEMM device time at that size hides the
    # dispatcher-channel call_boxed overhead.
    return Batch * M * N * K >= (1 << 29) and M >= 512 and N >= 512 and K >= 512


def _select_bmm_config(Batch, M, N, K, dtype):
    # gcu400: matrix_power_bmm has no entry in gcu400 tune_configs.yaml, so libtuner's
    # autotune falls back to the nvidia default small tile (TILE<=128x128x32, num_stages=2,
    # with a mask on every iteration); square GEMMs measured 3.5~11.7x slower than native. Switched here to
    # deterministic
    # tile dispatch, calibrated by scanning each level with libra L600 do_bench(200ms/100ms median):
    #   fp16/bf16 large square matrices (128,256,512,w2): optimal across m=1024/2048/4096;
    #   fp32 large square matrices (256,128,128,w2): ~10% faster than (128,256,128,w2);
    #   m<=512 small square matrices (64,64,256,w4): prioritize 48 SP occupancy (at m=384, 36 vs 6
    #   programs, fp32 reaches 0.60x native); fp64 operands are 8 bytes, so keep a small tile.
    if dtype == torch.float64:
        block_m, block_n, block_k, num_warps, num_stages = 64, 64, 32, 4, 3
    elif K < 128:
        # num_stages must be 3: ns=2 combined with bf16 triggers a gcu compiler llvm-link
        # segfault (tops_matrix_store_bf16_fp32_m32 serialization crash, rc=139).
        block_m, block_n, block_k, num_warps, num_stages = 64, 64, 32, 4, 3
    elif M <= 512 and N <= 512 and K <= 512:
        block_m, block_n, block_k, num_warps, num_stages = 64, 64, 256, 4, 3
    elif dtype == torch.float32:
        block_m, block_n, block_k, num_warps, num_stages = 256, 128, 128, 2, 2
    else:
        block_m, block_n, block_k, num_warps, num_stages = 128, 256, 512, 2, 2

    # Exact decision: when all three dimensions are divisible by the selected tile, compile an unmasked fast path.
    even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    return block_m, block_n, block_k, num_warps, num_stages, even


@libentry()
@triton.jit
def bmm_kernel(
    A,
    B,
    O,
    Batch,
    M,
    N,
    K,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n

    pid_b = pid // tiles_per_batch
    pid_mn = pid % tiles_per_batch

    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        A
        + pid_b * stride_ab
        + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        B
        + pid_b * stride_bb
        + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    # EVEN is a compile-time constant: a runtime branch degrades dot to scalar compute, so it must be trimmed
    # statically.
    if EVEN:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
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
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    o_ptrs = (
        O
        + pid_b * stride_ob
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )
    o = acc.to(O.dtype.element_ty)
    if EVEN:
        tl.store(o_ptrs, o)
    else:
        mask_o = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(o_ptrs, o, mask=mask_o)


def _launch_bmm(A, B, out, Batch, M, N, K):
    block_m, block_n, block_k, num_warps, num_stages, even = _select_bmm_config(
        Batch, M, N, K, A.dtype
    )
    grid = (Batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    dot_out_dtype = tl.float64 if A.dtype == torch.float64 else tl.float32
    bmm_kernel[grid](
        A,
        B,
        out,
        Batch,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dot_out_dtype=dot_out_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        EVEN=even,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _bmm(A, B, out=None, delegate=False):
    """Batched matrix multiply for contiguous 3D tensors of shape (batch, M, K) x (batch, K, N).

    ``delegate`` is decided once per power chain in _matrix_power (every chain
    product shares the same shape/dtype, and the per-call predicate costs a
    measurable slice of the launch-bound small shapes). Saturated products run
    on the import-time captured native GEMM kernels (call_boxed, bit-identical
    to the torch production path, ~zero dispatch overhead); everything else
    launches the vendored Triton kernel below with the deterministic tile
    dispatch.
    """
    if delegate:
        if A.shape[0] == 1:
            if out is None:
                return _NATIVE_MM.call_boxed(_DEVICE_KEYSET, A[0], B[0]).unsqueeze(0)
            if _NATIVE_MM_OUT is not None:
                _NATIVE_MM_OUT.call_boxed(_DEVICE_KEYSET, A[0], B[0], out=out[0])
                return out
        else:
            if out is None:
                return _NATIVE_BMM.call_boxed(_DEVICE_KEYSET, A, B)
            if _NATIVE_BMM_OUT is not None:
                return _NATIVE_BMM_OUT.call_boxed(_DEVICE_KEYSET, A, B, out=out)
    batch, M, K = A.shape
    N = B.shape[2]
    if out is None:
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
    _launch_bmm(A, B, out, batch, M, N, K)
    return out


def _matrix_power(A, n, dest=None):
    """Compute A ** n for a batched square matrix using the Triton bmm kernel.

    ``A`` is expected to be a contiguous 3D tensor of shape (batch, M, M).
    When ``dest`` is given (contiguous, non-aliasing), the final bmm writes
    its result directly into it, saving one full-matrix copy.
    """
    batch, M, _ = A.shape

    if n == 0:
        eye = torch.eye(M, dtype=A.dtype, device=A.device)
        eye = eye.unsqueeze(0).expand(batch, M, M)
        if dest is not None:
            dest.copy_(eye)
            return dest
        return eye.contiguous()

    if n < 0:
        # A ** (-k) == inv(A) ** k. Matrix inversion has no efficient Triton
        # equivalent here, so obtain the inverse through an LU factorization
        # and a solve against the identity.
        eye = (
            torch.eye(M, dtype=A.dtype, device=A.device)
            .unsqueeze(0)
            .expand(batch, M, M)
            .contiguous()
        )
        lu, pivots = torch.linalg.lu_factor(A)
        A = torch.linalg.lu_solve(lu, pivots, eye).contiguous()
        n = -n

    if n == 1:
        if dest is not None:
            dest.copy_(A)
            return dest
        return A.clone()

    # Binary exponentiation (repeated squaring) via batched matmuls. Count the
    # launches up front so the final bmm can write straight into ``dest``.
    # The chain's GEMM backend is decided once: every product shares A's
    # shape/dtype, so A itself is representative.
    delegate = _should_delegate_to_native(A, A)
    total_bmm = (n.bit_length() - 1) + (bin(n).count("1") - 1)
    launched = 0
    result = None
    base = A
    while n > 0:
        if n & 1:
            if result is None:
                result = base
            else:
                launched += 1
                result = _bmm(
                    result,
                    base,
                    out=dest if launched == total_bmm else None,
                    delegate=delegate,
                )
        n >>= 1
        if n > 0:
            launched += 1
            base = _bmm(
                base,
                base,
                out=dest if launched == total_bmm else None,
                delegate=delegate,
            )
    return result


def _validate_matrix_power_args(A):
    assert A.ndim >= 2, "matrix_power expects a tensor of at least 2 dimensions"
    assert A.shape[-1] == A.shape[-2], "matrix_power expects square matrices"
    assert A.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ], f"matrix_power only supports float16, bfloat16, float32, float64, got {A.dtype}"


def _dest_writable(out, A):
    # The final bmm reads its operand buffers while writing ``dest``; reject
    # any ``out`` that could overlap the input storage, plus shape/dtype/
    # contiguity mismatches, so the copy-based fallback stays available.
    if out.shape != A.shape or out.dtype != A.dtype or out.device != A.device:
        return False
    if not out.is_contiguous():
        return False
    if out.data_ptr() == A.data_ptr():
        return False
    if out.untyped_storage().data_ptr() == A.untyped_storage().data_ptr():
        return False
    return True


def matrix_power(A, n):
    logger.debug("GEMS_ENFLAME MATRIX_POWER")
    _validate_matrix_power_args(A)

    M = A.shape[-1]
    batch_shape = A.shape[:-2]
    A3 = A.reshape(-1, M, M).contiguous()

    out = _matrix_power(A3, n)
    return out.reshape(*batch_shape, M, M)


def matrix_power_out(A, n, *, out):
    logger.debug("GEMS_ENFLAME MATRIX_POWER_OUT")
    _validate_matrix_power_args(A)

    if _dest_writable(out, A):
        M = A.shape[-1]
        A3 = A.reshape(-1, M, M).contiguous()
        _matrix_power(A3, n, dest=out.reshape(-1, M, M))
        return out

    result = matrix_power(A, n)
    out.copy_(result)
    return out
