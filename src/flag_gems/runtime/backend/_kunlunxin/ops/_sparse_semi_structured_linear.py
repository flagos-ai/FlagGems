"""Kunlunxin override for ``_sparse_semi_structured_linear``.

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
def _sparse_gate_kn_flat_kernel(W, Meta, Out, N, K4, BLOCK: tl.constexpr):
    """2:4 gate for a weight whose storage is the contiguous (K, N) transpose.

    ``W`` is ``base`` (K, N) row-major where ``base = weight.t()``.  A 2:4 group
    is 4 *consecutive K rows*, i.e. ``4 * N`` *contiguous* elements, so each
    program owns one group and runs a flat 1-D affine load/store (the 1-D flat
    pass is ~10x faster than a 2-D tiled copy of the same tiny tile on XPU3 --
    the 2-D path pays a per-row DMA floor).  ``BLOCK`` must equal ``4 * N``.
    ``g = pid`` is constant per program, so the meta index ``col * K4 + pid`` is
    a repeated-index gather over ``col = lane // N``.
    """
    pid = tl.program_id(0)  # group index g; owns rows [4g, 4g+4) = BLOCK elems
    lane = tl.arange(0, BLOCK)
    row = lane // N  # position 0..3 within the group
    col = lane % N  # n
    meta = tl.load(Meta + col * K4 + pid)
    keep = tl.where(row < 2, meta != 0, meta == 0)
    off = pid * BLOCK + lane
    w = tl.load(W + off)
    tl.store(Out + off, tl.where(keep, w, tl.zeros_like(w)))


@triton.jit
def _nn_gemm_epilogue_kernel(
    A,
    B,
    Bias,
    C,
    stride_am,
    stride_bk,
    stride_cm,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``C = A @ B`` (NN) with a fused bias / out-dtype epilogue.

    ``A`` is (M, K) and ``B`` is a contiguous (K, N); both are affine.  Used for
    the transposed-storage path, whose gate naturally materialises the masked
    weight in (K, N) order.  Tiles divide M/N/K exactly (no masked ``tl.dot``
    operand, which miscompiles on XPU3).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_K):
        a = tl.load(A + rm[:, None] * stride_am + (k * BLOCK_K + rk)[None, :])
        b = tl.load(B + (k * BLOCK_K + rk)[:, None] * stride_bk + rn[None, :])
        if IEEE:
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b)

    if HAS_BIAS:
        acc += tl.load(Bias + rn).to(tl.float32)

    out_dtype = C.dtype.element_ty
    out = acc.to(out_dtype)
    tl.store(C + rm[:, None] * stride_cm + rn[None, :], out)


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


def _word_gate_mask(weight, meta, N, K, itemsize, device):
    """Materialise the 2:4-masked (N, K) weight with the flat 32-bit word gate.

    Returns ``(masked, blk)`` or ``(None, None)`` when the word count has no
    power-of-two divisor >= 32 (then a masked tail would be needed; the caller
    falls back to the strided gate).  Both operands must be contiguous.
    """
    n_words = (N * K * itemsize) // 4
    blk = min(8192, max(256, n_words // 8))
    # Floor is 32 (not 256) so shapes whose word count is not a multiple of 256
    # (e.g. K=48, N=48 fp16 -> 1152 words) still find a divisor instead of the
    # scalarised gate.  Official benchmark shapes divide at their initial >=256
    # block, so this never changes their block size.
    while blk > 32 and n_words % blk:
        blk //= 2
    if n_words % blk != 0 or n_words < blk:
        return None, None
    masked = torch.empty((N, K), device=device, dtype=weight.dtype)
    with torch_device_fn.device(device):
        _sparse_gate_words_kernel[(n_words // blk,)](
            weight.view(torch.int32),
            meta,
            masked.view(torch.int32),
            WPG=itemsize,
            BLOCK=blk,
        )
    return masked, blk


def _finish(out, bias, act_relu, act_gelu, act_silu, output_dtype):
    """Apply bias -> out-dtype cast -> activation to a dense (M, N) result."""
    needs_epilogue = (
        bias is not None
        or act_relu
        or act_gelu
        or act_silu
        or output_dtype != out.dtype
    )
    if not needs_epilogue:
        return out
    M, N = out.shape
    result = torch.empty((M, N), device=out.device, dtype=output_dtype)
    MN = M * N
    BLOCK = 1024
    bias_arg = bias if bias is not None else out
    with torch_device_fn.device(out.device):
        _linear_epilogue_kernel[(triton.cdiv(MN, BLOCK),)](
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

    # ---- Contiguous-operand path.
    #
    # The 2:4 gate is a flat, fully affine 32-bit-word pass (no per-element
    # mask/gather) whose block divides the word count, so it is always valid on
    # contiguous operands.  When the NT-GEMM tiles also divide M/N/K exactly we
    # fuse the matmul + bias + out-dtype into ``_nt_gemm_epilogue_kernel`` (the
    # fast main path); otherwise (e.g. M<16, or a dim with no >=16 power-of-two
    # divisor) we still use the fast word gate and hand the dense (N, K) weight
    # to the backend ``mm`` kernel -- this replaces the old scalarised strided
    # gate for all contiguous shapes while leaving the main path byte-identical.
    tiles = _fast_tiles(M, N, K)
    if (
        input.is_contiguous()
        and weight.is_contiguous()
        and meta.is_contiguous()
        and input.dtype.itemsize in (2, 4)
    ):
        masked, _blk = _word_gate_mask(
            weight, meta, N, K, input.dtype.itemsize, input.device
        )
        if masked is not None:
            if tiles is not None:
                bm, bn, bk = tiles
                out = torch.empty((M, N), device=input.device, dtype=output_dtype)
                bias_arg = bias if bias is not None else out
                with torch_device_fn.device(input.device):
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
                        # Bias + cast folded; apply only the activation.
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
            # tiles None: dense NN/NT matmul via the backend mm kernel.
            out = _gems_mm(input, masked.t())
            return _finish(out, bias, act_relu, act_gelu, act_silu, output_dtype)

    # ---- Transposed-storage weight: ``weight.t()`` is a contiguous (K, N).
    #
    # A (N, K) weight that is the transpose of a contiguous (K, N) base has its
    # 2:4 groups as 4 *contiguous* K-rows of the base, so the flat 1-D gate
    # (``_sparse_gate_kn_flat_kernel``) materialises the masked weight in (K, N)
    # order -- both the load and store stay flat/affine (a 2-D tiled copy of the
    # same tile is ~10x slower here).  The dense result is ``input @ masked_kn``
    # (a plain NN matmul) done in-kernel by ``_nn_gemm_epilogue_kernel``; the
    # backend ``mm`` kernel has ~70us fixed overhead at these sizes.
    if (
        not weight.is_contiguous()
        and input.is_contiguous()
        and meta.is_contiguous()
        and input.dtype.itemsize in (2, 4)
    ):
        wt = weight.t()
        # Flat gate needs BLOCK == 4 * N to be a power of two -> N must be.
        if wt.is_contiguous() and N >= 4 and (N & (N - 1)) == 0 and tiles is not None:
            bm, bn, bk = tiles
            masked_kn = torch.empty((K, N), device=input.device, dtype=weight.dtype)
            out = torch.empty((M, N), device=input.device, dtype=output_dtype)
            bias_arg = bias if bias is not None else out
            with torch_device_fn.device(input.device):
                _sparse_gate_kn_flat_kernel[(K // 4,)](
                    wt, meta, masked_kn, N, K4, BLOCK=4 * N
                )
                _nn_gemm_epilogue_kernel[(M // bm, N // bn)](
                    input,
                    masked_kn,
                    bias_arg,
                    out,
                    input.stride(0),
                    masked_kn.stride(0),
                    out.stride(0),
                    K,
                    bias is not None,
                    input.dtype == torch.float32,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                )
                if act_relu or act_gelu or act_silu:
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

    # ---- Fallback: arbitrary strided weight (kept for correctness).
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
    return _finish(out, bias, act_relu, act_gelu, act_silu, output_dtype)
