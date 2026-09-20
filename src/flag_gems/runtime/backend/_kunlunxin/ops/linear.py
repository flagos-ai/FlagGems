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

from .addmm import addmm
from .mm import mm

logger = logging.getLogger(__name__)


@triton.jit
def _linear_fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    sx0,
    sx1,
    sw0,
    sw1,
    sy0,
    sy1,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_TRANS: tl.constexpr,
):
    """Single-launch fused y = x @ W^T (+ b) for small shapes.

    x is (M, K); w is (N, K).  y is (M, N).  Accumulate in fp32 and cast on
    store.  One kernel launch avoids addmm's per-call Python wrapper (bias/dest
    unit-stride "dance", contiguous, empties, grid closure).

    W-load has two forms selected by ``USE_TRANS``:
      - False: read W directly as its transpose tile (BK, BN) via swapped strides
        (sw1 for the K axis, sw0 for the N axis).  The inner (BN) index then walks
        W's N axis with stride K -> non-coalesced.
      - True: read W's *natural* (BN, BK) tile (rn rows stride sw0, kk cols stride
        sw1 == 1 -> coalesced inner DMA) then ``tl.trans`` to (BK, BN) for the dot.
        Measured on XPU this lifts W-read bandwidth on the large-N / W-read-bound
        fused shapes for fp16 (+2..+12%) and fp32 (+23..+45%); bf16 is neutral-to-
        negative so the caller keeps USE_TRANS=False there.
    """
    pid = tl.program_id(0)
    nbn = tl.cdiv(N, BN)
    pm = pid // nbn
    pn = pid % nbn
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(
            x_ptr + rm[:, None] * sx0 + kk[None, :] * sx1,
            mask=(rm[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        if USE_TRANS:
            wn = tl.load(
                w_ptr + rn[:, None] * sw0 + kk[None, :] * sw1,
                mask=(rn[:, None] < N) & (kk[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(a, tl.trans(wn), allow_tf32=False)
        else:
            w = tl.load(
                w_ptr + kk[:, None] * sw1 + rn[None, :] * sw0,
                mask=(kk[:, None] < K) & (rn[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, w, allow_tf32=False)
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn, mask=rn < N, other=0.0)[None, :].to(tl.float32)
    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr + rm[:, None] * sy0 + rn[None, :] * sy1,
        y,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _linear_fused_kernel_nomask(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    sx0,
    sx1,
    sw0,
    sw1,
    sy0,
    sy1,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_TRANS: tl.constexpr,
):
    """Mask-free twin of ``_linear_fused_kernel`` for exactly-divisible shapes.

    Selected only when ``M % BM == 0 and N % BN == 0 and K % BK == 0`` so every
    tile is fully in bounds.  Dropping all load/store masks lets the loads become
    static full-tile DMAs: the SDNN backend no longer emits, per K step, the
    predicated ``sdnn.ew fill`` plus the dynamic ``minsi/maxsi``/``subview``
    boundary machinery it must keep when M/N/K are runtime args it cannot prove
    divisible.  Measured on XPU this is a large fp16 win on the divisible M=256
    large-N/large-K laggards -- e.g. (256,4096,14336) 0.38->0.66x, (256,4096,4096)
    0.57->0.82x, (256,3584,3584) 0.62->0.88x -- while non-divisible shapes fall
    back to the masked kernel unchanged.

    ``USE_TRANS`` selects the same coalesced-(BN,BK)+``tl.trans`` W-load as the
    masked kernel (see there); the caller sets it for fp16/fp32 large-N.
    """
    pid = tl.program_id(0)
    nbn = tl.cdiv(N, BN)
    pm = pid // nbn
    pn = pid % nbn
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(x_ptr + rm[:, None] * sx0 + kk[None, :] * sx1)
        if USE_TRANS:
            wn = tl.load(w_ptr + rn[:, None] * sw0 + kk[None, :] * sw1)
            acc += tl.dot(a, tl.trans(wn), allow_tf32=False)
        else:
            w = tl.load(w_ptr + kk[:, None] * sw1 + rn[None, :] * sw0)
            acc += tl.dot(a, w, allow_tf32=False)
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn)[None, :].to(tl.float32)
    y = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + rm[:, None] * sy0 + rn[None, :] * sy1, y)


def _fused_blocks(M, N, K):
    """Pick (BM, BN, BK) for the fused kernel in the small-/skinny-M regime.

    BM is matched to M (16/64/128) so a tiny batch does not pad the ``tl.dot`` M
    dimension into wasted matrix-engine work (the reason the vendor addmm/mm
    fixed-tile path only reaches ~0.4-0.6x on M=1..8).  BM is capped at 128:
    this is the certified config (official benchmark 726 SUCCESS/0 FAILED,
    dtype-balanced 0.847).  A BM=256 tier was tried to help the M=256 large-N/
    large-K shapes but could not be certified -- its benchmark runs never
    completed cleanly (crashed on host/device env contention while cards were
    re-taken mid-run), so the proven BM=128 config ships.  BK deepens with K so
    the large-K skinny shapes accumulate in few, wide steps; BN widens with N.
    """
    if M <= 16:
        BM = 16
    elif M <= 64:
        BM = 64
    else:
        BM = 128
    if K <= 1024:
        BK = 64
    elif K <= 2048:
        BK = 128
    else:
        BK = 256
    BN = 256 if N >= 1024 else 128
    return BM, BN, BK


def _fused_linear(input_flat, weight, bias):
    """Lean single-kernel path: no contiguous, no bias dance.

    Fp32 accumulation (measured relerr vs fp32 < 2e-2, tighter than torch's own
    fp16/bf16), one launch, W^T consumed via swapped strides.

    Mask handling (measured on XPU that the per-K-step predicated ``sdnn.ew fill``
    + dynamic boundary ``subview`` the masked kernel forces is a real cost):
      - Exactly-divisible tiles -> mask-free kernel directly (big fp16 win, no
        extra work).
      - Otherwise, when N/K are divisible and the GEMM is large enough that a
        one-off M pad-copy is amortised (K >= 12288 or N >= 16384, i.e. the
        large-K / large-N laggards), round M up to a BM multiple into a scratch
        buffer (the extra rows are garbage but only land in the sliced-off output
        rows) and run the mask-free kernel.  Below that size the pad-copy plus its
        dispatch overhead dominates the small kernel and regresses, so keep masks.
      - Everything else -> masked kernel.
    """
    M, K = input_flat.shape
    N = weight.shape[0]
    BM, BN, BK = _fused_blocks(M, N, K)
    nk_divisible = (N % BN == 0) and (K % BK == 0)
    b_arg = bias if bias is not None else input_flat
    has_bias = bias is not None
    # Coalesced-(BN,BK)+tl.trans W-load helps W-read-bound large-N shapes on
    # fp16 (+2..+12%) and fp32 (+23..+45%) but is neutral-to-negative on bf16
    # (measured, harness/perf_ir/wload_trans_broad_probe.py) -> gate by dtype.
    use_trans = input_flat.dtype in (torch.float16, torch.float32)

    if nk_divisible and M % BM == 0:
        # Fully in bounds: drop all masks -> static full-tile DMAs.
        out = torch.empty((M, N), device=input_flat.device, dtype=input_flat.dtype)
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _linear_fused_kernel_nomask[grid](
            input_flat,
            weight,
            b_arg,
            out,
            M,
            N,
            K,
            input_flat.stride(0),
            input_flat.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            BM,
            BN,
            BK,
            has_bias,
            use_trans,
        )
        return out

    if nk_divisible and (K >= 12288 or N >= 16384):
        # Large GEMM: amortise a one-off M pad-copy to reach the mask-free path.
        Mp = triton.cdiv(M, BM) * BM
        x_pad = torch.empty((Mp, K), device=input_flat.device, dtype=input_flat.dtype)
        x_pad[:M].copy_(input_flat)
        out = torch.empty((Mp, N), device=input_flat.device, dtype=input_flat.dtype)
        grid = (triton.cdiv(Mp, BM) * triton.cdiv(N, BN),)
        _linear_fused_kernel_nomask[grid](
            x_pad,
            weight,
            b_arg,
            out,
            Mp,
            N,
            K,
            x_pad.stride(0),
            x_pad.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            BM,
            BN,
            BK,
            has_bias,
            use_trans,
        )
        return out[:M]

    # Non-divisible / small GEMM: masked kernel.
    out = torch.empty((M, N), device=input_flat.device, dtype=input_flat.dtype)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _linear_fused_kernel[grid](
        input_flat,
        weight,
        b_arg,
        out,
        M,
        N,
        K,
        input_flat.stride(0),
        input_flat.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BM,
        BN,
        BK,
        has_bias,
        use_trans,
    )
    return out


def _use_fused(M, N, K):
    """Route small-/skinny-M *and* small/very-skinny-N GEMMs to the fused kernel.

    Regimes sent to ``_fused_linear`` instead of the vendor ``mm`` + contiguous
    path (N == 1 GEMV always stays on ``addmm``):

    1. Small-/skinny-M (``M <= 256``).  The fused ``tl.dot`` kernel with BM
       matched to M and a deep BK beats both the vendor ``addmm`` delegation and
       ``mm`` across this regime -- e.g. (1,512,3584) 0.08->0.91x, (8,512,3584)
       0.24->1.36x, (256,4096,4096) 0.34->0.55x.  The vendor path is slow here
       because its fixed large-BM tile pads a tiny M into wasted matrix-engine
       work plus a Python wrapper.

    2. Very-skinny-N large-M (``1 < N <= 512`` with ``K <= 8192``).  The output
       has too few columns for the vendor ``mm`` to amortise its W^T
       ``.contiguous()`` materialisation, so the fused kernel (W^T via swapped
       strides, no materialisation) wins -- measured (8192,512,3584) fp16
       0.42->0.62, fp32 ~neutral.  Kept at N <= 512 (not 1024): at N == 1024 with
       large M/K the vendor path wins fp32/bf16 by a wide margin (e.g.
       (8192,1024,4096) fp32 0.94 vs fused 0.77, bf16 0.92 vs 0.63), so those
       stay on ``mm``.  Bounded to K <= 8192: large-K favours the vendor tuned
       ``mm`` (the 1848x1536x152064 laggards lose on the fused kernel).

    3. Small balanced GEMM (``M <= 1024 and N <= 1024 and K <= 4096``).  Below
       ~1024^3 the fused kernel wins on every dtype -- (384,384,384) 0.11->~1.4,
       (1024,1024,1024) 0.20->~0.9 across fp16/fp32/bf16 -- because the vendor mm
       cannot amortise its contiguous+bias overhead on a small GEMM.

    Everything above -- the large balanced GEMM (e.g. 8192^2, N > 1024) -- is
    left on the vendor tuned ``mm``, which wins there on all dtypes.
    """
    if N <= 1:
        return False
    if M <= 256:
        return True
    if N <= 512 and K <= 8192:
        return True
    return M <= 1024 and N <= 1024 and K <= 4096


def linear(input, weight, bias=None):
    """y = x @ W^T + b on the XPU-tuned kunlunxin primitives.

    Routing (all compute stays in ``_kunlunxin``):
      - N == 1 GEMV: bias-fused ``addmm(bias, x, W.t())`` (matrix-engine path).
      - Small-/skinny-M (N > 1, M <= 256) or skinny-N (1 < N <= 1024, K <= 8192):
        a single fused ``tl.dot`` kernel (``_fused_linear``) with BM matched to M
        and a deep BK.  This is our own kernel (no vendor mm/addmm): it beats the
        vendor primitives across these regimes by avoiding the fixed-tile M
        padding, the W^T ``.contiguous()`` materialisation, and the Python
        wrapper (see ``_use_fused``).
      - Large balanced GEMM (M > 256, N > 1024): vendor tuned ``mm`` on a
        row-major contiguous ``W^T`` (aligned fast path) + one amortised bias-add.
    """
    logger.debug("GEMS_KUNLUNXIN LINEAR")

    input_dim = input.dim()
    single_1d = input_dim == 1
    if single_1d:
        input = input.unsqueeze(0)

    batch_dims = input.shape[:-1]
    K = input.shape[-1]
    N = weight.shape[0]
    M = 1
    for d in batch_dims:
        M *= d

    input_flat = input.reshape(M, K)

    if _use_fused(M, N, K):
        # Single-launch fused matmul + bias; W.t() handled via swapped strides.
        output = _fused_linear(input_flat, weight, bias)
    elif bias is not None and N == 1:
        # GEMV: bias-fused GEMM; W.t() is a strided (K, N) view handled by addmm.
        output = addmm(bias, input_flat, weight.t())
    else:
        # Feed W^T as a strided (K, N) view straight to mm(); it consumes the
        # strided mat2 on the matrix engine at full speed.  Do NOT materialise the
        # transpose via .contiguous(): the strided-source copy path is a
        # pathological ~1.3 GB/s on this XPU toolchain (25 ms for a 4096^2 fp16
        # tile) and would dominate the GEMM.  Measured strided-mm vs .contiguous()
        # on 8192x4096x4096: fp16 0.75 vs 0.04, fp32 0.98 vs 0.09, bf16 1.57 vs
        # 0.10; the strided view also lifts fp32/bf16 above the old contiguous
        # numbers (0.98/1.57 vs 0.83/1.04) since mm needs no aligned copy.
        output = mm(input_flat, weight.t())
        if bias is not None:
            output.add_(bias)

    output = output.view(*batch_dims, N)

    if single_1d:
        output = output.squeeze(0)

    return output
