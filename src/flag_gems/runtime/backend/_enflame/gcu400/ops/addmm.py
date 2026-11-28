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
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry

logger = logging.getLogger(__name__)

GCU_MAX_GRID_Y = 255

# Reusable empty context: when the device already matches, it replaces the construction/destruction overhead of
# torch_device_fn.device(...)
_NULL_CONTEXT = nullcontext()
# Tile-dispatch cache: _select_addmm_config is a pure function, saving ~2us per call on the host side
_CONFIG_CACHE = {}

# ---------------------------------------------------------------------------
# Native-kernel delegation (capture + call_boxed), same pattern as the local
# mm / bmm / router_gemm / linalg_multi_dot ops. The kernels are captured at
# import time - before any use_gems() registration replaces the PrivateUse1
# entry - so call_boxed runs the original device kernel bit-identically with
# zero dispatch overhead (bmm precedent: verified 1.000-1.006x vs plain
# torch.bmm). Dispatch line mirrors mm (single-sourced threshold):
#   M*N*K >= 2^29 (saturated segment) or K >= 2^15 (long-K skinny segment).
# Measured anchors (mm campaign, kernel mode do_bench median, this board):
#   384^3  Triton 1.59/1.75x (fp16/fp32) stays on the Triton path
#   768^3  Triton 1.13/1.14x stays on the Triton path
#   1024^3 native wins (Triton 0.94/0.93/0.70x) - delegated
#   2048^3 native wins (0.69x), 4096^3 native wins (0.68-0.76x) - delegated
# Baseline addmm campaign (2026-11-16, core bench): 1024^3/2048^3/4096^3 sat
# at 0.66-0.96x across dtypes - exactly the native-wins segment.
# ---------------------------------------------------------------------------
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_ADDMM = None
_NATIVE_ADDMM_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_ADDMM = torch.library.get_kernel(
            torch.ops.aten.addmm.default, runtime.device.dispatch_key
        )
        _NATIVE_ADDMM_OUT = torch.library.get_kernel(
            torch.ops.aten.addmm.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_ADDMM = None
        _NATIVE_ADDMM_OUT = None

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)
_NATIVE_ADDMM_WORK_THRESHOLD = 1 << 29
_ADDMM_LONG_K_THRESHOLD = 1 << 15


def _should_delegate_to_native(bias, mat1, mat2):
    if _NATIVE_ADDMM is None:
        return False
    if mat1.device.type != runtime.device.name:
        return False
    M, K = mat1.shape
    N = mat2.shape[1]
    return M * N * K >= _NATIVE_ADDMM_WORK_THRESHOLD or K >= _ADDMM_LONG_K_THRESHOLD


def _select_addmm_config(M, N, K, dtype):
    """Deterministically picks the GEMM tile from shape and dtype; avoids autotune cross-dtype misjudgement.

    Large-matrix arm (2026-11-16 addmm perf round, the Triton tier outside the delegated segment):
      - 16-bit: BK=512 single-segment tile covers the whole band with K<=512 (384^3 class),
        per the concurrent mm/router_gemm measurements, winner (128,128,512,w2,s2) 384^3 7.4us
        (this round's same-shape baseline 13.8us); K>512 keeps K%256==0 -> 256 half-depth blocks;
        num_warps 4 -> 2 across all tiers (linear/tensordot precedent: GCU400 large tiles prefer w2
        over w4, 2048^3 same tile w4 750us -> w2 -15%). DSM budget (128,128,512,16bit) is
        ~320KB < the 448KB two-program line.
      - fp32: when M/N/K are divisible use the 64-row small tile (64,128,128) (linear same-source
        scan measured 384^3 fp32 11.2us 1.59x, this round's baseline 15.3us); otherwise keep the
        128-row large block.
    """
    if K <= 64 or M <= 64 or N <= 64:
        if K <= 64 or N <= 16:
            # Short K / very narrow N: a small tile suffices, avoiding lots of masked-out useless computation
            block_m, block_n, block_k = 32, 64, 32 if K <= 32 else 64
            num_warps = 4
            num_stages = 3
        else:
            # Small M/N but long K on the other side: BK=64 inflates the tile iteration count by K/64; measured
            # 64x384x7168 at 175us; switching to a deep block of 256 + a narrow column block of 64 gives 56-60us.
            # Keeping the column block at 64
            # still fills enough programs when N is only 128~512 (BN=128 was measured to regress
            # to 108us at N=384).
            block_m, block_n = 64, 64
            block_k = 128 if K <= 128 else 256
            num_warps = 4
            num_stages = 2
    else:
        block_m = 128
        # When N is within 512, a column block of 128 evenly divides common shapes, enabling the EVEN fast path
        block_n = 128 if N <= 512 else 256
        if dtype != torch.float32:
            if K <= 128:
                block_k = 128
            elif K <= 512:
                block_k = 512
            elif K % 256 == 0:
                block_k = 256
            else:
                block_k = 128
            num_warps = 2
            num_stages = 2
        else:
            if M % 64 == 0 and N % 128 == 0 and K % 128 == 0:
                block_m, block_n, block_k = 64, 128, 128
            else:
                block_k = 128
            num_warps = 2
            num_stages = 3

    grid_n = triton.cdiv(N, block_n)
    if grid_n > GCU_MAX_GRID_Y:
        block_n = triton.cdiv(N, GCU_MAX_GRID_Y)
        block_n = max(triton.next_power_of_2(block_n), 16)

    return block_m, block_n, block_k, num_warps, num_stages


def _classify_bias(bias, m, n):
    """Route the bias to a 1-D or scalar addressing arm whenever a broadcast
    stride collapses to zero, and materialize nothing on the host.

    The matrix arm must never receive a zero stride: a 0-stride scalar used in
    the 2-D bias-load addressing lowers to an out-of-bounds access that trips a
    device RAS interrupt and kills the process (one-shot repro in
    compiler_issues, addmm zero-stride bias load). Returns
    (bias, stride_im, stride_in, is_vector, is_scalar, is_colvec).
    """
    if bias.numel() == 1:
        return bias, 0, 0, False, True, False
    if bias.ndim == 1:
        return bias, 0, bias.stride(0), True, False, False
    # Common fast path: bias shape exactly matches out with non-zero strides, so skip broadcast_to
    if bias.shape[0] == m and bias.shape[1] == n:
        bias_stride_m, bias_stride_n = bias.stride()
        if bias_stride_m != 0 and bias_stride_n != 0:
            return bias, bias_stride_m, bias_stride_n, False, False, False
    bias_view = bias.broadcast_to((m, n))
    stride_im = bias_view.stride(0)
    stride_in = bias_view.stride(1)
    if stride_im == 0:
        return bias, 0, stride_in, True, False, False
    if stride_in == 0:
        return bias, stride_im, 0, False, False, True
    return bias_view, stride_im, stride_in, False, False, False


def _select_addmm_out_config(M, N, K, dtype):
    """Tile dispatch for addmm_out; out is supplied by the caller.

    2026-11-16 perf round consolidated this onto _select_addmm_config (linalg_matmul criterion:
    a sibling variant building its own tile table is a structural drift source): the saturated
    segment is delegated to native and the Triton segment only keeps the small/medium shapes
    addmm also serves, where the base config measured better than the old override tiers --
    384^3 16-bit old (128,256,512,w2,s2) 10.2-11.1us vs base (128,128,512,w2,s2) 7.9-8.9us
    (BN=256 costs +29%); fp32 384^3 15.1us vs 13.5us. Kept only because addmm_dtype/addmm_ import its symbol.
    """
    return _select_addmm_config(M, N, K, dtype)


def _cached_config(config_fn, M, N, K, dtype):
    """Caches the tile dispatch result by (config_fn, M, N, K, dtype).

    Dispatch itself is a pure function, and a cache hit saves ~2us/call of host time; on host-bound
    small shapes (device only 10-30us) that is a real share of the latency.
    """
    key = (config_fn, M, N, K, dtype)
    config = _CONFIG_CACHE.get(key)
    if config is None:
        config = config_fn(M, N, K, dtype)
        _CONFIG_CACHE[key] = config
    return config


@libentry()
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
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
    BIAS_IS_VECTOR: tl.constexpr = False,
    BIAS_IS_SCALAR: tl.constexpr = False,
    BIAS_IS_COLVEC: tl.constexpr = False,
    SKIP_BIAS: tl.constexpr = False,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN:
            # When all three dimensions are divisible by the tile, no out-of-bounds elements exist, so compile time
            # keeps only the unmasked dot pipeline
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
                other=0.0,
            )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # PyTorch ignores bias, including NaN and Inf values, when beta is zero.
    if SKIP_BIAS:
        accumulator = accumulator * alpha
    else:
        if BIAS_IS_VECTOR:
            if EVEN:
                bias = tl.load(i_ptr + offs_cn * stride_in)[None, :]
            else:
                bias = tl.load(
                    i_ptr + offs_cn * stride_in, mask=offs_cn < N, other=0.0
                )[None, :]
        elif BIAS_IS_COLVEC:
            if EVEN:
                bias = tl.load(i_ptr + offs_cm * stride_im)[:, None]
            else:
                bias = tl.load(
                    i_ptr + offs_cm * stride_im, mask=offs_cm < M, other=0.0
                )[:, None]
        elif BIAS_IS_SCALAR:
            bias = tl.load(i_ptr)
        else:
            i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
            if EVEN:
                bias = tl.load(i_ptrs)
            else:
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                bias = tl.load(i_ptrs, mask=c_mask, other=0.0)
        accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(c_ptr.dtype.element_ty)
    if EVEN:
        tl.store(c_ptrs, c)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_n1_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    stride_im,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SKIP_BIAS: tl.constexpr = False,
    BIAS_IS_SCALAR: tl.constexpr = False,
):
    """Dedicated reduction kernel for N == 1 (matrix-vector product).

    With N == 1 the generic GEMM tile has a single valid column and the 2-D masked dot/store
    degrades to a per-column element-wise path (measured 1x1x2048 1.71ms, 2048x1x1024 5.57ms);
    this kernel drops the N dim entirely: each program owns BLOCK_SIZE_M rows, accumulates an
    fp32 reduction vector along K, and keeps only 1-D masked accesses.

    BIAS_IS_SCALAR covers bias as a scalar / (1,1) / (1,) expansion (column broadcast); every
    other legal N==1 bias shape is (M,1), addressed by row stride_im.
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = offs_m < M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k_block * BLOCK_SIZE_K + offs_k
        k_mask = k_offs < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(b_ptr + k_offs * stride_bk, mask=k_mask, other=0.0)
        acc += tl.sum(a.to(tl.float32) * b[None, :].to(tl.float32), axis=1)

    if SKIP_BIAS:
        acc = acc * alpha
    else:
        if BIAS_IS_SCALAR:
            bias_value = tl.load(i_ptr).to(tl.float32)
        else:
            bias_value = tl.load(i_ptr + offs_m * stride_im, mask=m_mask, other=0.0).to(
                tl.float32
            )
        acc = acc * alpha + bias_value * beta

    tl.store(
        c_ptr + offs_m * stride_cm,
        acc.to(c_ptr.dtype.element_ty),
        mask=m_mask,
    )


def _select_addmm_n1_config(M, K):
    """Blocking for the N == 1 reduction kernel: parallel over row blocks, deep K blocks to saturate bandwidth."""
    block_m = 64 if M >= 64 else max(triton.next_power_of_2(M), 16)
    block_k = 256 if K >= 256 else max(triton.next_power_of_2(K), 16)
    return block_m, block_k, 4, 2


def _launch_addmm_n1(bias, mat1, mat2, out, beta, alpha):
    """N == 1 branch: goes through addmm_n1_kernel, avoiding the generic GEMM tile column degradation."""
    M, K = mat1.shape
    bias, bias_sm, bias_sn, is_vector, is_scalar, is_colvec = _classify_bias(bias, M, 1)
    # When N == 1, the column direction (is_vector) can only reach column 0, equivalent to scalar broadcast.
    bias_is_scalar = is_scalar or is_vector

    block_m, block_k, num_warps, num_stages = _select_addmm_n1_config(M, K)
    grid = (triton.cdiv(M, block_m),)
    with torch_device_fn.device(mat1.device):
        addmm_n1_kernel[grid](
            mat1,
            mat2,
            bias,
            out,
            alpha,
            beta,
            M,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            out.stride(0),
            bias_sm,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_K=block_k,
            SKIP_BIAS=bool(beta == 0),
            BIAS_IS_SCALAR=bias_is_scalar,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def _launch_addmm(bias, mat1, mat2, out, beta, alpha, config_fn):
    M, K = mat1.shape
    _, N = mat2.shape

    if N == 1:
        return _launch_addmm_n1(bias, mat1, mat2, out, beta, alpha)

    stride_am, stride_ak = mat1.stride()
    stride_bk, stride_bn = mat2.stride()
    if stride_am > 1 and stride_ak > 1:
        mat1 = mat1.contiguous()
        stride_am, stride_ak = mat1.stride()
    if stride_bk > 1 and stride_bn > 1:
        mat2 = mat2.contiguous()
        stride_bk, stride_bn = mat2.stride()

    bias, bias_sm, bias_sn, is_vector, is_scalar, is_colvec = _classify_bias(bias, M, N)

    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _cached_config(
        config_fn, M, N, K, mat1.dtype
    )
    even = (M % BLOCK_M == 0) and (N % BLOCK_N == 0) and (K % BLOCK_K == 0)

    # A non-row-major out (e.g. a .t() view) under "masked + row-contiguous" vectorized stores would only write
    # the tile's first row; here we fall back to the safe path: write into a contiguous temporary buffer, then copy back
    # to out.
    store_out = out
    need_copy_back = out.stride(1) != 1
    if need_copy_back:
        store_out = torch.empty((M, N), device=out.device, dtype=out.dtype)
    stride_cm, stride_cn = store_out.stride()

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    # When the current device already matches, skip one context switch (~2-3us/call, same as arange)
    device_scope = (
        _NULL_CONTEXT
        if mat1.device.index == torch_device_fn.current_device()
        else torch_device_fn.device(mat1.device)
    )
    with device_scope:
        addmm_kernel[grid](
            mat1,
            mat2,
            bias,
            store_out,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            bias_sm,
            bias_sn,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_M=8,
            EVEN=even,
            BIAS_IS_VECTOR=is_vector,
            BIAS_IS_SCALAR=is_scalar,
            BIAS_IS_COLVEC=is_colvec,
            SKIP_BIAS=bool(beta == 0),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if need_copy_back:
            out.copy_(store_out)
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_ENFLAME ADDMM")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    if bias.numel() != 1 and bias.shape != (mat1.shape[0], mat2.shape[1]):
        # Common fast path (bias shape already equals out) skips one broadcastable_to call
        assert broadcastable_to(
            bias.shape, (mat1.shape[0], mat2.shape[1])
        ), "Incompatible input shape"
    # Saturated / long-K segment: delegate to the captured native kernel.
    # Bit-identical to the torch.addmm production path (the native kernel
    # broadcasts the bias, honors beta==0 NaN/Inf semantics and takes
    # arbitrary strides directly - no .contiguous() copies), with ~zero
    # dispatch overhead (mm/bmm precedent).
    if _should_delegate_to_native(bias, mat1, mat2):
        return _NATIVE_ADDMM.call_boxed(
            _DEVICE_KEYSET, bias, mat1, mat2, beta=beta, alpha=alpha
        )
    # new_empty reuses an existing tensor's device/dtype, saving one
    # device/dtype resolution versus torch.empty(device=...) (this round's A/B: 85us -> 81us)
    out = mat1.new_empty((mat1.shape[0], mat2.shape[1]))
    return _launch_addmm(bias, mat1, mat2, out, beta, alpha, _select_addmm_config)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_ENFLAME ADDMM_OUT")
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    if bias.numel() != 1 and bias.shape != (mat1.shape[0], mat2.shape[1]):
        # Common fast path (bias shape already equals out) skips one broadcastable_to call
        assert broadcastable_to(
            bias.shape, (mat1.shape[0], mat2.shape[1])
        ), "Incompatible input shape"
    if out is None:
        out = mat1.new_empty((mat1.shape[0], mat2.shape[1]))
    else:
        assert out.shape == (mat1.shape[0], mat2.shape[1]), "Incompatible output shape"
    # Native out= delegation keeps a strict fast-case guard (mm_out
    # precedent): uniform dtypes and a contiguous out. Non-contiguous out
    # (the .t()-view segment handled by the temp-buffer path below) and
    # mixed-dtype calls keep the pre-campaign Triton behavior.
    if (
        _should_delegate_to_native(bias, mat1, mat2)
        and _NATIVE_ADDMM_OUT is not None
        and out.is_contiguous()
        and out.dtype == mat1.dtype
        and bias.dtype == mat1.dtype
        and mat2.dtype == mat1.dtype
    ):
        return _NATIVE_ADDMM_OUT.call_boxed(
            _DEVICE_KEYSET, bias, mat1, mat2, beta=beta, alpha=alpha, out=out
        )
    return _launch_addmm(bias, mat1, mat2, out, beta, alpha, _select_addmm_out_config)
