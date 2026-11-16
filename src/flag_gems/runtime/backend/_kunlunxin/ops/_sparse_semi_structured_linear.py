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
"""Kunlunxin XPU3 override for ``_sparse_semi_structured_linear``.

The generic kernel computes its 2:4 selection on a rank-3 tile and reduces it
with ``tl.sum(selected, axis=2)`` inside a runtime ``for k in range(num_iters)``
loop.  On TritonXPU that fails to lower (``TritonXPULegalize`` -> "Failures have
been detected while processing an MLIR pass pipeline"), so the op is completely
unusable on this backend (26/26 tests fail at compile time).

This override keeps every tile 1-D/2-D and never reduces: it reuses the two
patterns already proven on XPU3 elsewhere in this backend:

1. a 2-D elementwise "gate" kernel that materialises the 2:4-masked weight
   (same technique as ``_sparse_semi_structured_mm._sparse_gate_kernel``);
   the mask is resolved per column so the kernel issues exactly one masked
   load and one masked store (an unrolled 4-store gate drops sibling stores
   when the tail mask is partial -- see the ``rrelu_`` pattern);
2. the backend's battle-tested ``mm`` kernel for the dense matmul;
3. an optional flat 1-D epilogue that folds bias / out-dtype cast / activation
   with a single masked store.

All masked stores clamp their addresses into the allocation (a masked store
whose inactive addresses leave the buffer corrupts active lanes / faults).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

from .mm import mm as _gems_mm

logger = logging.getLogger(__name__)

erf = tl_extra_shim.erf


@triton.jit
def _sparse_gate_linear_kernel(
    W,
    Meta,
    O,
    N,
    K4,
    stride_wn,
    stride_wk,
    stride_mn,
    stride_mk,
    stride_on,
    stride_ok,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """O[n, c] = W[n, c] if column c is kept by the 2:4 meta else 0.

    ``c`` is a full K column index (K = 4 * K4); its group is ``c // 4`` and its
    within-group position ``c % 4`` decides which meta value keeps it
    (positions 0,1 kept when ``meta != 0``; positions 2,3 when ``meta == 0``).
    Resolving the keep predicate per column keeps this to one load + one store.
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    K = 4 * K4
    mask = (offs_n[:, None] < N) & (offs_c[None, :] < K)

    # Clamp inactive lanes into the allocation: masked stores/loads whose
    # addresses leave the buffer mis-lower on XPU3.
    nc = tl.minimum(offs_n, N - 1)
    cc = tl.minimum(offs_c, K - 1)
    grp = offs_c // 4
    gc = tl.minimum(grp, K4 - 1)
    pos = offs_c % 4

    meta = tl.load(
        Meta + nc[:, None] * stride_mn + gc[None, :] * stride_mk,
        mask=mask,
        other=0,
    )
    w = tl.load(
        W + nc[:, None] * stride_wn + cc[None, :] * stride_wk,
        mask=mask,
        other=0.0,
    )
    keep = tl.where(pos[None, :] < 2, meta != 0, meta == 0)
    val = tl.where(keep, w, tl.zeros_like(w))
    tl.store(
        O + nc[:, None] * stride_on + cc[None, :] * stride_ok,
        val,
        mask=mask,
    )


@triton.jit
def _linear_epilogue_kernel(
    X,
    Bias,
    Out,
    MN,
    N,
    HAS_BIAS: tl.constexpr,
    ACT_RELU: tl.constexpr,
    ACT_GELU: tl.constexpr,
    ACT_SILU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold bias -> out-dtype cast -> activation over the (M, N) output."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    # Clamp inactive lanes (and the bias gather index) into the allocation.
    oc = tl.minimum(offs, MN - 1)
    nc = oc % N

    x = tl.load(X + oc, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    if HAS_BIAS:
        b = tl.load(Bias + nc, mask=mask, other=0.0).to(tl.float32)
        xf = xf + b

    out_dtype = Out.dtype.element_ty
    out = xf.to(out_dtype)

    if ACT_RELU:
        out = tl.where(out > 0, out, tl.zeros_like(out))
    elif ACT_GELU:
        xg = out.to(tl.float32)
        scale: tl.constexpr = 0.7071067811  # 1 / sqrt(2)
        out = (0.5 * xg * (1.0 + erf(xg * scale))).to(out_dtype)
    elif ACT_SILU:
        xs = out.to(tl.float32)
        # tl.fdiv keeps this from being canonicalised into ``tt.silu``, which
        # the XPU backend's EW-table pass cannot lower (KeyError: 'tt.silu').
        out = tl.fdiv(xs, 1.0 + tl.exp(-xs)).to(out_dtype)

    tl.store(Out + oc, out, mask=mask)


@triton.jit
def _sparse_gate_words_kernel(W32, Meta, O32, WPG: tl.constexpr, BLOCK: tl.constexpr):
    """Fast 2:4 gate over 32-bit words (flat 1-D, every access affine).

    A 2:4 group is four consecutive elements, i.e. ``WPG = itemsize`` 32-bit
    words (2 for fp16/bf16, 4 for fp32).  Word ``p`` of a group belongs
    to the kept pair iff ``(p < WPG // 2) == (meta != 0)``, which is constant
    across the whole 32-bit word, so the gate is a bitwise AND with all-ones.

    ``g = lane // WPG`` is a *repeated* index into the contiguous meta array, so
    both the meta load and the data load/store stay affine/contiguous (no
    per-lane data-dependent gather, which would scalarise the kernel on XPU3).
    """
    pid = tl.program_id(0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    g = j // WPG
    p = j % WPG
    meta = tl.load(Meta + g)
    keep = (p < WPG // 2) == (meta != 0)
    w = tl.load(W32 + j)
    tl.store(O32 + j, tl.where(keep, w, tl.zeros_like(w)))


@triton.jit
def _nt_gemm_epilogue_kernel(
    A,
    B,
    Bias,
    C,
    stride_am,
    stride_bn,
    stride_cm,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``C = A @ B.T`` with a fused bias / out-dtype / activation epilogue.

    ``A`` is (M, K) and ``B`` is (N, K), both K-contiguous, so the second
    operand is consumed with ``tl.dot(a, tl.trans(b))`` (NT GEMM) -- measured
    ~10 us on XPU3 for 256x512 versus ~1.1 ms for the element-wise gate path.

    Strictly 2-arg ``tl.dot`` (``acc += tl.dot(...)``; the 3-arg form
    miscompiles) on tiles that tile M, N and K exactly, so no operand is ever
    value-masked (masked dot operands do not lower with correct values on XPU3).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_K):
        a = tl.load(A + rm[:, None] * stride_am + (k * BLOCK_K + rk)[None, :])
        b = tl.load(B + rn[:, None] * stride_bn + (k * BLOCK_K + rk)[None, :])
        if IEEE:
            # fp32: the default dot truncates to TF32 (max err ~8e-2 on K=512);
            # ieee is fp32-exact and measured the same cost on XPU3.
            acc += tl.dot(a, tl.trans(b), input_precision="ieee")
        else:
            acc += tl.dot(a, tl.trans(b))

    if HAS_BIAS:
        acc += tl.load(Bias + rn).to(tl.float32)

    # Only bias + out-dtype cast are folded here.  Activations are applied by
    # ``_linear_epilogue_kernel`` afterwards: a kernel that contains ``tl.dot``
    # is lowered by the TTSDNN pipeline, whose EW-activation LUT pass cannot
    # lower the ``tt.silu`` / ``tt.gelu`` ops that the normaliser emits
    # (``KeyError: 'tt.silu'`` at compile time).
    out_dtype = C.dtype.element_ty
    out = acc.to(out_dtype)
    tl.store(C + rm[:, None] * stride_cm + rn[None, :], out)


def _pow2_div(n: int, cap: int) -> int:
    """Largest power of two that divides ``n`` and is <= ``cap``."""
    b = 1
    while b * 2 <= cap and n % (b * 2) == 0:
        b *= 2
    return b


def _fast_tiles(M: int, N: int, K: int):
    """Tile shapes for the NT-GEMM fast path, or None when it does not apply.

    ``tl.dot`` needs BLOCK_M/BLOCK_N/BLOCK_K >= 16 and (on XPU3) the operands
    must never be value-masked, so every tile has to divide its dimension.
    """
    bm = _pow2_div(M, 128)
    bn = _pow2_div(N, 128)
    bk = _pow2_div(K, 64)
    if bm < 16 or bn < 16 or bk < 16:
        return None
    return bm, bn, bk


def _sparse_semi_structured_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    meta: torch.Tensor,
    bias: torch.Tensor = None,
    activation: str = None,
    out_dtype: torch.dtype = None,
):
    """2:4 sparse semi-structured linear on XPU3: ``input @ weight.T (+ bias)``.

    ``weight`` is (N, K) and ``meta`` is (N, K // 4).  See the generic
    implementation for the full 2:4 contract; the semantics here are identical.
    """
    logger.debug("GEMS_KUNLUNXIN SPARSE_SEMI_STRUCTURED_LINEAR")

    M, K = input.shape
    N, K_w = weight.shape
    K4 = K // 4

    assert K == K_w, f"Incompatible dimensions: input K={K}, weight K={K_w}"
    assert K % 4 == 0, f"K must be a multiple of 4 for 2:4 sparsity, got K={K}"
    assert meta.shape == (N, K4), f"Expected meta shape ({N}, {K4}), got {meta.shape}"
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ), f"Unsupported dtype: {input.dtype}"

    output_dtype = out_dtype if out_dtype is not None else input.dtype

    act = activation.lower() if activation is not None else None
    act_relu = act == "relu"
    act_gelu = act == "gelu"
    act_silu = act in ("silu", "swish")
    if activation is not None and not (act_relu or act_gelu or act_silu):
        raise ValueError(
            f"Unsupported activation: {activation!r}. "
            "Expected one of: 'relu', 'gelu', 'silu' (or 'swish'), or None."
        )

    # ---- Fast path: flat word gate + single NT-GEMM with fused epilogue.
    #
    # Requires contiguous operands (so the gate is a flat affine pass and both
    # GEMM operands are K-contiguous) and tiles that divide M/N/K exactly (so
    # no ``tl.dot`` operand has to be masked, which miscompiles on XPU3).
    tiles = _fast_tiles(M, N, K)
    if (
        tiles is not None
        and input.is_contiguous()
        and weight.is_contiguous()
        and meta.is_contiguous()
        and input.dtype.itemsize in (2, 4)
    ):
        bm, bn, bk = tiles
        # A group is 4 elements = 4 * itemsize bytes = ``itemsize`` 32-bit words.
        wpg = input.dtype.itemsize
        n_words = (N * K * input.dtype.itemsize) // 4
        # Aim for ~8 programs; BLOCK must divide the word count (no masks).
        blk = min(8192, max(256, n_words // 8))
        while blk > 256 and n_words % blk:
            blk //= 2
        if n_words % blk == 0 and n_words >= blk:
            masked = torch.empty((N, K), device=input.device, dtype=weight.dtype)
            out = torch.empty((M, N), device=input.device, dtype=output_dtype)
            bias_arg = bias if bias is not None else out
            with torch_device_fn.device(input.device):
                _sparse_gate_words_kernel[(n_words // blk,)](
                    weight.view(torch.int32),
                    meta,
                    masked.view(torch.int32),
                    WPG=wpg,
                    BLOCK=blk,
                )
                _nt_gemm_epilogue_kernel[(M // bm, N // bn)](
                    input,
                    masked,
                    bias_arg,
                    out,
                    input.stride(0),
                    masked.stride(0),
                    out.stride(0),
                    K,
                    bias is not None,
                    input.dtype == torch.float32,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                )
                if act_relu or act_gelu or act_silu:
                    # Bias + cast are already folded; apply only the activation.
                    result = torch.empty_like(out)
                    mn = M * N
                    eblk = 1024
                    _linear_epilogue_kernel[(triton.cdiv(mn, eblk),)](
                        out,
                        out,
                        result,
                        mn,
                        N,
                        False,
                        act_relu,
                        act_gelu,
                        act_silu,
                        BLOCK=eblk,
                    )
                    return result
            return out

    # 1) Materialise the 2:4-masked weight, row-major (N, K).
    masked = torch.empty((N, K), device=input.device, dtype=weight.dtype)
    BLOCK_N = 16
    BLOCK_C = 512
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_C))
    with torch_device_fn.device(input.device):
        _sparse_gate_linear_kernel[grid](
            weight,
            meta,
            masked,
            N,
            K4,
            weight.stride(0),
            weight.stride(1),
            meta.stride(0),
            meta.stride(1),
            masked.stride(0),
            masked.stride(1),
            BLOCK_N=BLOCK_N,
            BLOCK_C=BLOCK_C,
        )

    # 2) Dense matmul with the backend ``mm`` kernel: input (M, K) @ masked.T.
    out = _gems_mm(input, masked.t())

    # 3) Optional fused epilogue (bias -> cast -> activation).
    needs_epilogue = (
        bias is not None
        or act_relu
        or act_gelu
        or act_silu
        or output_dtype != input.dtype
    )
    if not needs_epilogue:
        return out

    result = torch.empty((M, N), device=input.device, dtype=output_dtype)
    MN = M * N
    BLOCK = 1024
    egrid = (triton.cdiv(MN, BLOCK),)
    bias_arg = bias if bias is not None else out
    with torch_device_fn.device(input.device):
        _linear_epilogue_kernel[egrid](
            out,
            bias_arg,
            result,
            MN,
            N,
            bias is not None,
            act_relu,
            act_gelu,
            act_silu,
            BLOCK=BLOCK,
        )
    return result
