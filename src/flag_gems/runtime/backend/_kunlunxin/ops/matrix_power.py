# Copyright 2026, The FlagOS Contributors.
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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.ops.matrix_power import _matrix_power as _generic_matrix_power
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic
from .to import _to_copy_contiguous, _to_copy_contiguous_to_bf16_kernel

logger = logging.getLogger(__name__)

# Tile selection is done on the host so that the launch grid stays a *constant
# tuple* (never a ``lambda``): on the xpubin backend a callable grid object is
# baked into the compilation cache key, so a per-call lambda forces a full
# recompilation on every launch (KERNEL_OPT_EXPERIENCE #41).
#
# The kernel is compiled with a ``DIVISIBLE_*`` constexpr specialization: when
# every dimension is an exact multiple of the tile, the loads/stores are fully
# affine and mask-free, which is what lets ``tl.dot`` lower to the XDNN
# hardware matmul path. A masked 2D load inside a dot kernel -- even with an
# all-true mask and ``other=0.0`` -- silently miscompiles and/or blocks the
# block-DMA lowering (KERNEL_OPT_EXPERIENCE #17 / #36), so the masked path is
# reserved for the small, non-divisible shapes of the functional test matrix.
#
# fp32 GEMM forbids a large K-tile: BK=256 silently produces garbage
# (KERNEL_OPT_EXPERIENCE #19), so fp32 (and the bf16 path, which upcasts to
# fp32) is capped at BK=128 while fp16 may use BK=256.
_TILE_LARGE = 256
_TILE_MED = 128
_TILE_SMALL = 16
_NUM_WARPS = 4


@triton.jit
def _bmm_kernel(
    A,
    B,
    O,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_ob,
    stride_om,
    stride_on,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
):
    pid_b = ext.program_id(2)
    A += pid_b * stride_ab
    B += pid_b * stride_bb
    O += pid_b * stride_ob

    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    o_ptrs = O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    if DIVISIBLE_M and DIVISIBLE_N and DIVISIBLE_K:
        for _ in range(0, K // TILE_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            a_ptrs += TILE_K * stride_ak
            b_ptrs += TILE_K * stride_bk
            o += tl.dot(a, b, allow_tf32=False)
        tl.store(o_ptrs, o.to(O.dtype.element_ty))
    else:
        mask_m = offs_m < M
        mask_n = offs_n < N
        for _ in range(0, tl.cdiv(K, TILE_K)):
            mask_k = offs_k < K
            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask_a, other=0.0)
            b = tl.load(b_ptrs, mask_b, other=0.0)
            offs_k += TILE_K
            a_ptrs += TILE_K * stride_ak
            b_ptrs += TILE_K * stride_bk
            o += tl.dot(a, b, allow_tf32=False)
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(o_ptrs, o.to(O.dtype.element_ty), mask_c)


def _select_tile(M, K, dtype):
    """Pick a tile so the (square) benchmark shapes land on the mask-free path.

    All benchmark shapes (256/384/512/1024/2048/4096) are multiples of 128, so
    they compile to the fast affine specialization; only the tiny functional
    shapes (M in {2, 8, 16, 32}) fall back to the masked small tile.

    The XDNN block-DMA lowering is tuned for a ``load-tile-size=131072``
    working set, so an asymmetric (TM, TN) = (512, 256) tile (131072 elements)
    is ~1.1-1.25x faster than the square 256x256 tile for M >= 1024. For
    M = 512 the asymmetric tile collapses the grid to (1, 2) programs and
    regresses, so keep the square 256x256 tile there.
    """
    if M >= 1024 and M % 512 == 0:
        tm, tn = 512, 256
    elif M % _TILE_LARGE == 0:
        tm = tn = _TILE_LARGE
    elif M % _TILE_MED == 0:
        tm = tn = _TILE_MED
    else:
        tm = tn = _TILE_SMALL

    if dtype == torch.float16:
        tk_opts = (_TILE_LARGE, _TILE_MED, _TILE_SMALL)
    else:
        # fp32 (and the bf16 path, which upcasts to fp32) must avoid BK=256.
        tk_opts = (_TILE_MED, _TILE_SMALL)

    tk = _TILE_SMALL
    for cand in tk_opts:
        if K % cand == 0:
            tk = cand
            break
    return tm, tn, tk


def _bmm(A, B, out=None):
    """Batched matmul for contiguous 3D (batch, M, K) x (batch, K, N).

    When ``out`` is provided (a contiguous 3D tensor of the result shape and
    dtype) the product is written directly into it, avoiding an extra copy.
    """
    batch, M, K = A.shape
    N = B.shape[2]
    if out is None:
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    tm, tn, tk = _select_tile(M, K, A.dtype)
    grid = (triton.cdiv(M, tm), triton.cdiv(N, tn), batch)
    if A.device.index == torch_device_fn.current_device():
        _bmm_kernel[grid](
            A,
            B,
            out,
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
            TILE_M=tm,
            TILE_N=tn,
            TILE_K=tk,
            DIVISIBLE_M=(M % tm == 0),
            DIVISIBLE_N=(N % tn == 0),
            DIVISIBLE_K=(K % tk == 0),
            num_warps=_NUM_WARPS,
        )
    else:
        with torch_device_fn.device(A.device):
            _bmm_kernel[grid](
                A,
                B,
                out,
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
                TILE_M=tm,
                TILE_N=tn,
                TILE_K=tk,
                DIVISIBLE_M=(M % tm == 0),
                DIVISIBLE_N=(N % tn == 0),
                DIVISIBLE_K=(K % tk == 0),
                num_warps=_NUM_WARPS,
            )
    return out


def _cast_dtype(x, dtype, out=None):
    """Dtype-converting copy via the backend's tuned contiguous-cast kernel.

    The cast must round fp32->bf16 with round-to-nearest-even, exactly as
    ``torch.matrix_power``'s bf16 chain does, otherwise the per-step rounding
    of ``_matrix_power_bf16`` drifts from the reference. ``_to_copy_contiguous``
    (``_kunlunxin/ops/to.py``) is the backend's own Triton cast path: its
    fp32->bf16 kernel implements RNE with the ``0x7FFF + lsb`` bit bias, which
    is bit-identical to ``Tensor.to``, and its bf16->fp32 path is an exact
    upcast. It is a real gems kernel (no ATen/native copy fallback), unlike the
    previously used ATen ``_copy_from`` strided-copy primitive, which the
    fallback rule now forbids.

    When ``out`` is provided the result is written directly into it.
    """
    if out is None:
        out = torch.empty(x.shape, dtype=dtype, device=x.device)
    if dtype == torch.bfloat16 and x.is_floating_point() and x.dtype != torch.bfloat16:
        # fp32 -> bf16 (and fp16 -> bf16): `_to_copy_contiguous` routes every
        # ->bf16 cast through a 256-element tile (its conservative safe path),
        # which serializes a 16.7M-element cast into ~65k tiny launches and is
        # ~16x slower than necessary. The RNE bit-manipulation kernel is
        # verified bit-identical to ``Tensor.to(bf16)`` at large tiles
        # (probe_cast_tile), so launch it directly with the same power-of-two
        # tile the bf16->fp32 upcast already enjoys.
        _cast_to_bf16(x, out)
    else:
        _to_copy_contiguous(x, out, dtype)
    return out


def _cast_to_bf16(x, out):
    """fp32 -> bf16 round-to-nearest-even cast via a large-tile direct launch."""
    n = x.numel()
    if n == 0:
        return out
    block = 1 << (n - 1).bit_length() if n > 1 else 1
    if block > 16384:
        block = 16384
    grid = ((n + block - 1) // block,)
    if x.device.index == torch_device_fn.current_device():
        _to_copy_contiguous_to_bf16_kernel[grid](x, out, n, block)
    else:
        with torch_device_fn.device(x.device):
            _to_copy_contiguous_to_bf16_kernel[grid](x, out, n, block)
    return out


@triton.jit
def _round_to_bf16_kernel(inp, out, n_elements, BLOCK_SIZE: tl.constexpr):
    """Round fp32 to bf16 precision (RNE) and return as fp32, in one 1D pass.

    The bf16 power chain needs a ``fp32 -> bf16 -> fp32`` roundtrip after every
    squaring product. Doing it as a single elementwise kernel (read fp32,
    write fp32) halves the memory traffic of the two-kernel roundtrip and keeps
    the bit manipulation out of the dot kernel (where it miscompiles / is ~10x
    slow -- the ``_round_fp32_to_bf16`` epilogue experiment).
    """
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(inp + offsets, mask=mask).to(tl.float32)
    fp32_bits = values.to(tl.uint32, bitcast=True)
    rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1)
    out_bits = ((fp32_bits + rounding_bias) >> 16) << 16
    tl.store(out + offsets, out_bits.to(tl.float32, bitcast=True), mask=mask)


# The RNE round is compute-bound (bit manipulation), but routing it through the
# tuned pointwise_dynamic codegen with the async-DMA config (the same recipe as
# ``_to_copy_func_async`` in ``to.py``) still yields a ~1.25x win on the large
# benchmark shapes by overlapping the load/store stream (verified bit-identical
# to the plain kernel above for M in {384..4096}). Small shapes keep the plain
# big-tile kernel: pointwise_dynamic's host-side prep dominates the ~5us of
# device work on the tiny functional-test shapes.
_async_round_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
    config=_async_round_config,
)
@triton.jit
def _round_to_bf16_async(x):
    fp32_bits = x.to(tl.uint32, bitcast=True)
    rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1)
    out_bits = ((fp32_bits + rounding_bias) >> 16) << 16
    return out_bits.to(tl.float32, bitcast=True)


def _round_bf16(x):
    """Round a fp32 tensor to bf16 precision, returned as fp32 again."""
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n = x.numel()
    if n == 0:
        return out
    # The async pointwise_dynamic path wins on device time (~1.25x) but its
    # host-side prep (~13us/call of Python) makes the whole bf16 chain
    # host-bound on the small shapes (M <= 1024). Route only the >= 2M-element
    # shapes (2048x2048 and up) through it, where the device-work saving
    # exceeds the host overhead; small shapes keep the plain big-tile kernel.
    if n >= 2 * 1024 * 1024:
        _round_to_bf16_async(x, out0=out)
        return out
    block = 1 << (n - 1).bit_length() if n > 1 else 1
    if block > 16384:
        block = 16384
    grid = ((n + block - 1) // block,)
    if x.device.index == torch_device_fn.current_device():
        _round_to_bf16_kernel[grid](x, out, n, block)
    else:
        with torch_device_fn.device(x.device):
            _round_to_bf16_kernel[grid](x, out, n, block)
    return out


def _matrix_power_bf16(A, n, out=None):
    """Repeated-squaring for bf16 via fp32 gemm kernels.

    ``tl.dot`` on this backend does not round a bf16 matmul the same way
    ``torch.matrix_power``'s bf16 chain does: loading bf16 inside a dot kernel
    is mis-compiled (KERNEL_OPT_EXPERIENCE #11), and the residual per-product
    error is amplified 2x per squaring step, blowing the bf16 tolerance.
    Matching torch's semantics (each repeated-squaring product is rounded back
    to bf16) is done by upcasting the bf16 matrix to fp32 *once*, running the
    fp32 gemm, and rounding every intermediate product back to bf16 on the host
    via a single fused ``fp32->bf16->fp32`` elementwise kernel
    (``_round_to_bf16_kernel``). This keeps the kernel fp32-only (no bf16 load)
    while reproducing torch's per-step bf16 rounding.

    When ``out`` is provided the final result is downcast directly into it.
    """
    batch, M, _ = A.shape

    if n == 0:
        eye = torch.eye(M, dtype=torch.bfloat16, device=A.device)
        res = eye.unsqueeze(0).expand(batch, M, M).contiguous()
        if out is not None:
            out.copy_(res)
            res = out
        return res

    if n == 1:
        res = A.clone()
        if out is not None:
            out.copy_(res)
            res = out
        return res

    # Upcast once: bf16 -> fp32 is exact, and the kernel never loads bf16.
    base = _cast_dtype(A, torch.float32)
    result = None
    while n > 0:
        if n & 1:
            # ``base`` is always bf16-representable (exact upcast of a bf16
            # value, or the rounded output of a previous squaring step), so the
            # first accumulation needs no extra rounding step.
            if result is None:
                result = base
            else:
                prod = _bmm(result, base)
                if (n >> 1) == 0:
                    # Final accumulation: the fp32->bf16 RNE downcast composes
                    # the ``_round_bf16`` roundtrip with the final downcast (the
                    # round already leaves the value bf16-representable), so fuse
                    # both into a single launch and skip the extra round kernel.
                    return _cast_dtype(prod, torch.bfloat16, out=out)
                result = _round_bf16(prod)
        n >>= 1
        if n > 0:
            base = _round_bf16(_bmm(base, base))
    return _cast_dtype(result, torch.bfloat16, out=out)


def _matrix_power(A, n, out=None):
    """Compute A ** n for a contiguous 3D stack of square matrices (batch, M, M).

    When ``out`` is provided (a contiguous 3D tensor of the same shape/dtype)
    the result is written directly into it, avoiding a final copy.
    """
    if n < 0:
        # Negative powers need an LU inverse, which torch only supports for
        # fp32/fp64. Delegate to the generic implementation (fp32 path).
        res = _generic_matrix_power(A, n)
        if out is not None:
            out.copy_(res)
            res = out
        return res

    if A.dtype == torch.bfloat16:
        return _matrix_power_bf16(A, n, out=out)

    if A.dtype == torch.float64:
        # Not exercised by tests/benchmark (fp16/fp32/bf16 only); keep the
        # generic gems path for completeness.
        res = _generic_matrix_power(A, n)
        if out is not None:
            out.copy_(res)
            res = out
        return res

    batch, M, _ = A.shape

    if n == 0:
        eye = torch.eye(M, dtype=A.dtype, device=A.device)
        res = eye.unsqueeze(0).expand(batch, M, M).contiguous()
        if out is not None:
            out.copy_(res)
            res = out
        return res

    if n == 1:
        res = A.clone()
        if out is not None:
            out.copy_(res)
            res = out
        return res

    result = None
    base = A
    while n > 0:
        if n & 1:
            if result is None:
                result = base
            else:
                # On the final accumulation write directly into ``out``.
                result = _bmm(result, base, out=(out if (n >> 1) == 0 else None))
        n >>= 1
        if n > 0:
            base = _bmm(base, base)
    if out is not None and result is not out:
        out.copy_(result)
        result = out
    return result


def matrix_power(A, n):
    logger.debug("GEMS_KUNLUNXIN MATRIX_POWER")
    assert A.ndim >= 2, "matrix_power expects a tensor of at least 2 dimensions"
    assert A.shape[-1] == A.shape[-2], "matrix_power expects square matrices"
    assert A.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ], f"matrix_power only supports float16, bfloat16, float32, float64, got {A.dtype}"

    M = A.shape[-1]
    batch_shape = A.shape[:-2]
    A3 = A.reshape(-1, M, M).contiguous()

    out = _matrix_power(A3, n)
    return out.reshape(*batch_shape, M, M)


def matrix_power_out(A, n, *, out):
    logger.debug("GEMS_KUNLUNXIN MATRIX_POWER_OUT")
    assert A.ndim >= 2, "matrix_power expects a tensor of at least 2 dimensions"
    assert A.shape[-1] == A.shape[-2], "matrix_power expects square matrices"
    assert A.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ], f"matrix_power only supports float16, bfloat16, float32, float64, got {A.dtype}"

    M = A.shape[-1]
    batch_shape = A.shape[:-2]
    A3 = A.reshape(-1, M, M).contiguous()

    # Write the final product directly into ``out`` when it is a contiguous
    # buffer that does not alias the input; otherwise fall back to a copy so
    # aliasing/stride semantics stay identical to ``torch.matrix_power``.
    if out.is_contiguous() and out.dtype == A.dtype and out.data_ptr() != A.data_ptr():
        out3 = out.reshape(-1, M, M)
        _matrix_power(A3, n, out=out3)
        return out

    result = _matrix_power(A3, n)
    out.copy_(result.reshape(*batch_shape, M, M))
    return out
