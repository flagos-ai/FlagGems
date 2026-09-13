import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kunlunxin backend-local binary_cross_entropy.
#
# WHY THIS FILE EXISTS (correctness, not only speed):
#   The generic implementation (src/flag_gems/ops/binary_cross_entropy.py)
#   reduces with `tl.where(mask, vals, 0.0)` + a scalar `tl.atomic_add` over
#   BLOCK_SIZE=1024 tiles.  On this backend that kernel *cannot be compiled*:
#   the XPU triton backend keeps retuning the ELF stack size and finally
#   raises `RuntimeError: Failed to tune buffer size.` after ~200 s, so every
#   reduction='mean'/'sum' case fails (see
#   harness/solution/performance/binary_cross_entropy_xpu2_20260829.md).
#   reduction='none' compiles but runs at ~100 GB/s.
#
# DESIGN (all numbers measured on XPU 2, /tmp/fg_bce2_cfg_probe*.py):
#   * one unmasked BLOCK=32768 tile per CTA (`buffer_size_limit=2048` keeps
#     `tl.sum` inside its XPU correctness window, HARNESS_SUMMARY.md 2.5):
#       - fused reduce  146 GB/s fp16 / 288 GB/s fp32 (best correct config;
#         BLOCK=65536 and BLOCK=32768,U=2 silently lose half the lanes)
#       - flat/elementwise 256 GB/s fp16 / 499 GB/s fp32 (vs 104 / 270 GB/s
#         for the BLOCK=2048,U=8 tile the sibling with_logits op uses)
#   * masked loads are unreliable on this backend, so nothing that feeds a
#     reduction is ever masked.  Only whole unmasked tiles go through the
#     reduce kernel; the trailing partial tile is evaluated by the elementwise
#     kernel into a zero-filled fp32 scratch slot instead, because there a
#     wrong masked load is discarded by the masked store and can never be
#     observed.  (Copying the tail into a dtype-sized scratch tile first also
#     works but costs ~0.14 ms of extra launches per call.)
#   * the partials plus that scratch region are folded, normalised and cast by
#     one extra single-CTA kernel.  Going through `mid.sum() / N` instead costs
#     ~4 extra gems launches (~0.1 ms), which dominates every small/medium
#     shape.
#   * non-contiguous inputs keep the proven pointwise_dynamic path.
# ---------------------------------------------------------------------------

# tl.sum correctness window on this backend: BLOCK <= 8192 without
# `buffer_size_limit`, or BLOCK == 32768 together with buffer_size_limit=2048.
# 32768 is both the fastest and the safe point, so it is used everywhere.
_TILE = 32768
_BUF = 2048
# masked elementwise tail only (masked lanes are dropped by the masked store)
_TAIL_BLOCK = 2048
_TAIL_U = 8
_TAIL_SPAN = _TAIL_BLOCK * _TAIL_U
# single-CTA fold covers up to _TILE partials, i.e. N <= 2**30
_FOLD_BLOCK = _TILE


@triton.jit
def _bce_loss(xv, yv):
    # PyTorch clamps the log terms (not the input) at -100, so input==0 with
    # target==1 yields exactly 100 instead of inf.
    log_x = tl.maximum(tl.log(xv), -100.0)
    log_1mx = tl.maximum(tl.log(1.0 - xv), -100.0)
    return -(yv * log_x + (1.0 - yv) * log_1mx)


# ----------------------- fused reduction (unmasked only) --------------------


@triton.jit
def _bce_reduce_kernel(x, y, mid, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    xv = tl.load(x + idx).to(tl.float32)
    yv = tl.load(y + idx).to(tl.float32)
    tl.store(mid + pid, tl.sum(_bce_loss(xv, yv)))


@triton.jit
def _bce_weight_reduce_kernel(x, y, w, mid, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    xv = tl.load(x + idx).to(tl.float32)
    yv = tl.load(y + idx).to(tl.float32)
    wv = tl.load(w + idx).to(tl.float32)
    tl.store(mid + pid, tl.sum(_bce_loss(xv, yv) * wv))


@triton.jit
def _bce_fold_kernel(
    part, tail, out, DENOM, BLOCK: tl.constexpr, HAS_TAIL: tl.constexpr
):
    # Both buffers are zero-filled, so unused slots contribute nothing.  They
    # are summed separately because a single tl.sum is only reliable up to
    # BLOCK == 32768 on this backend.
    total = tl.sum(tl.load(part + tl.arange(0, BLOCK)))
    if HAS_TAIL:
        total += tl.sum(tl.load(tail + tl.arange(0, BLOCK)))
    tl.store(out, (total / DENOM).to(out.dtype.element_ty))


# ------------------------- flat kernels (reduction=none) --------------------


@triton.jit
def _bce_flat_kernel(
    x, y, out, N, BLOCK: tl.constexpr, U: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.5).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(out + idx, _bce_loss(xv, yv).to(out.dtype.element_ty), mask=m)
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            tl.store(out + idx, _bce_loss(xv, yv).to(out.dtype.element_ty))


@triton.jit
def _bce_weight_flat_kernel(
    x, y, w, out, N, BLOCK: tl.constexpr, U: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.5).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(
                out + idx, (_bce_loss(xv, yv) * wv).to(out.dtype.element_ty), mask=m
            )
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            wv = tl.load(w + idx).to(tl.float32)
            tl.store(out + idx, (_bce_loss(xv, yv) * wv).to(out.dtype.element_ty))


# ------------------ pointwise_dynamic path (non-contiguous) -----------------


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_pw_kernel(x, y):
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    log_x = tl.maximum(tl.log(xf), -100.0)
    log_1mx = tl.maximum(tl.log(1.0 - xf), -100.0)
    return -(yf * log_x + (1.0 - yf) * log_1mx)


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_weight_pw_kernel(x, y, w):
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    wf = w.to(tl.float32)
    log_x = tl.maximum(tl.log(xf), -100.0)
    log_1mx = tl.maximum(tl.log(1.0 - xf), -100.0)
    return -(yf * log_x + (1.0 - yf) * log_1mx) * wf


# --------------------------------- helpers ----------------------------------


def _normalize_reduction(reduction):
    # 0 = none, 1 = mean, 2 = sum
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _prepare_weight(weight, input, n_elements):
    # Returns a contiguous flat per-element weight tensor (or None).
    if weight is None:
        return None
    weight = weight.contiguous()
    if weight.numel() == 1 and n_elements != 1:
        return torch.full(
            (n_elements,), weight.item(), device=input.device, dtype=input.dtype
        )
    if weight.numel() != n_elements:
        raise AssertionError(
            "binary_cross_entropy: weight must have same number "
            "of elements as input and target, or be a scalar."
        )
    return weight.reshape(-1)


def _pw_elementwise(input, target, weight):
    if weight is None:
        return _bce_pw_kernel(input, target)
    return _bce_weight_pw_kernel(input, target, weight.reshape(input.shape))


def _pw_reduced(input, target, weight, red, n_elements):
    vals = _pw_elementwise(input, target, weight).to(torch.float32).reshape(-1)
    total = vals.sum()
    if red == 2:
        return total.to(input.dtype)
    return (total / float(n_elements)).to(input.dtype)


# ------------------------------- fast paths ---------------------------------


def _fast_elementwise(xf, yf, wf, n_elements, out):
    """Elementwise BCE over flat contiguous inputs, writing into flat `out`."""
    of = out.reshape(-1)
    n_tiles = n_elements // _TILE
    bulk = n_tiles * _TILE
    tail = n_elements - bulk
    with torch_device_fn.device(xf.device):
        if n_tiles:
            if wf is None:
                _bce_flat_kernel[(n_tiles,)](
                    xf,
                    yf,
                    of,
                    bulk,
                    BLOCK=_TILE,
                    U=1,
                    NEED_MASK=False,
                    buffer_size_limit=_BUF,
                )
            else:
                _bce_weight_flat_kernel[(n_tiles,)](
                    xf,
                    yf,
                    wf,
                    of,
                    bulk,
                    BLOCK=_TILE,
                    U=1,
                    NEED_MASK=False,
                    buffer_size_limit=_BUF,
                )
        if tail:
            grid = (triton.cdiv(tail, _TAIL_SPAN),)
            need_mask = (tail % _TAIL_SPAN) != 0
            if wf is None:
                _bce_flat_kernel[grid](
                    xf[bulk:],
                    yf[bulk:],
                    of[bulk:],
                    tail,
                    BLOCK=_TAIL_BLOCK,
                    U=_TAIL_U,
                    NEED_MASK=need_mask,
                )
            else:
                _bce_weight_flat_kernel[grid](
                    xf[bulk:],
                    yf[bulk:],
                    wf[bulk:],
                    of[bulk:],
                    tail,
                    BLOCK=_TAIL_BLOCK,
                    U=_TAIL_U,
                    NEED_MASK=need_mask,
                )
    return out


def _reduce_partials(xf, yf, wf, n_elements, n_slots):
    """Build the fp32 scratch that `_bce_fold_kernel` consumes.

    Layout: `[0, n_slots)` holds one partial per full unmasked BLOCK tile,
    `[n_slots, n_slots + _TILE)` holds the individual losses of the trailing
    partial tile.  The buffer is zero-filled, so unused slots are inert.

    The tail is handled by the *elementwise* kernel rather than by a reduce
    kernel: its masked lanes are discarded by the masked store (a wrong masked
    load can therefore never be observed), which avoids copying the tail into a
    dtype-sized scratch tile first (measured ~0.14 ms of extra launches per
    call, which dominated every shape whose numel is not a multiple of _TILE).
    """
    n_tiles = n_elements // _TILE
    bulk = n_tiles * _TILE
    tail = n_elements - bulk
    mid = torch.zeros(
        n_slots + (_TILE if tail else 0), dtype=torch.float32, device=xf.device
    )
    with torch_device_fn.device(xf.device):
        if n_tiles:
            if wf is None:
                _bce_reduce_kernel[(n_tiles,)](
                    xf, yf, mid, BLOCK=_TILE, buffer_size_limit=_BUF
                )
            else:
                _bce_weight_reduce_kernel[(n_tiles,)](
                    xf, yf, wf, mid, BLOCK=_TILE, buffer_size_limit=_BUF
                )
        if tail:
            grid = (triton.cdiv(tail, _TAIL_SPAN),)
            need_mask = (tail % _TAIL_SPAN) != 0
            if wf is None:
                _bce_flat_kernel[grid](
                    xf[bulk:],
                    yf[bulk:],
                    mid[n_slots:],
                    tail,
                    BLOCK=_TAIL_BLOCK,
                    U=_TAIL_U,
                    NEED_MASK=need_mask,
                )
            else:
                _bce_weight_flat_kernel[grid](
                    xf[bulk:],
                    yf[bulk:],
                    wf[bulk:],
                    mid[n_slots:],
                    tail,
                    BLOCK=_TAIL_BLOCK,
                    U=_TAIL_U,
                    NEED_MASK=need_mask,
                )
    return mid


def _fast_reduced(xf, yf, wf, red, n_elements, dtype, out):
    n_tiles = n_elements // _TILE
    denom = float(n_elements) if red == 1 else 1.0

    if n_tiles > _FOLD_BLOCK:
        # N > 2**30: too many partials for the single-CTA fold, so fold with
        # the (correct, but launch-heavier) gems sum instead.
        mid = _reduce_partials(xf, yf, wf, n_elements, n_tiles)
        result = (mid.sum() / denom).to(dtype)
        return result if out is None else _write_out(result, out)

    mid = _reduce_partials(xf, yf, wf, n_elements, _FOLD_BLOCK)
    has_tail = mid.numel() > _FOLD_BLOCK
    result = out if out is not None else torch.empty((), dtype=dtype, device=xf.device)
    with torch_device_fn.device(xf.device):
        _bce_fold_kernel[(1,)](
            mid,
            mid[_FOLD_BLOCK:] if has_tail else mid,
            result,
            denom,
            BLOCK=_FOLD_BLOCK,
            HAS_TAIL=has_tail,
            buffer_size_limit=_BUF,
        )
    return result


# --------------------------------- dispatch ---------------------------------


def _write_out(result, out):
    if result.data_ptr() == out.data_ptr():
        return out
    torch.ops.aten._copy_from(result.to(out.dtype).reshape(out.shape), out, False)
    return out


def _bce_dispatch(input, target, weight, red, out=None):
    if input.numel() != target.numel():
        raise AssertionError(
            "binary_cross_entropy: input and target must have the same number "
            "of elements."
        )
    n_elements = input.numel()
    weight = _prepare_weight(weight, input, n_elements)
    contiguous = (
        input.is_contiguous()
        and target.is_contiguous()
        and (weight is None or weight.is_contiguous())
    )

    if red == 0:
        if n_elements == 0:
            result = torch.empty_like(input)
        elif contiguous:
            buf = out
            if not (
                buf is not None
                and buf.is_contiguous()
                and buf.dtype == input.dtype
                and buf.numel() == n_elements
            ):
                buf = torch.empty_like(input)
            result = _fast_elementwise(
                input.reshape(-1), target.reshape(-1), weight, n_elements, buf
            )
        else:
            result = _pw_elementwise(input.contiguous(), target.contiguous(), weight)
    elif n_elements == 0:
        # PyTorch: sum -> 0, mean -> NaN
        if red == 2:
            result = torch.zeros((), device=input.device, dtype=input.dtype)
        else:
            result = torch.full(
                (), float("nan"), device=input.device, dtype=input.dtype
            )
    elif contiguous:
        scalar_out = out
        if not (
            scalar_out is not None
            and scalar_out.numel() == 1
            and scalar_out.dtype == input.dtype
            and scalar_out.is_contiguous()
        ):
            scalar_out = None
        result = _fast_reduced(
            input.reshape(-1),
            target.reshape(-1),
            weight,
            red,
            n_elements,
            input.dtype,
            scalar_out,
        )
    else:
        result = _pw_reduced(
            input.contiguous(), target.contiguous(), weight, red, n_elements
        )

    if out is None:
        return result
    return _write_out(result, out)


# --------------------------------- wrappers ---------------------------------


def binary_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction=1,
):
    logger.debug("GEMS_KUNLUNXIN BINARY_CROSS_ENTROPY")
    return _bce_dispatch(input, target, weight, _normalize_reduction(reduction))


def binary_cross_entropy_out(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    reduction=1,
    out: torch.Tensor = None,
):
    logger.debug("GEMS_KUNLUNXIN BINARY_CROSS_ENTROPY_OUT")
    red = _normalize_reduction(reduction)
    if out is None:
        return _bce_dispatch(input, target, weight, red)
    if red == 0:
        if out.numel() != input.numel():
            raise AssertionError(
                "binary_cross_entropy_out: for reduction='none', out must "
                "match input shape."
            )
    elif out.numel() != 1:
        raise AssertionError(
            "binary_cross_entropy_out: for reduction='sum' or 'mean', out must "
            "be a scalar tensor."
        )
    if out.device != input.device:
        raise AssertionError(
            "binary_cross_entropy_out: out must be on the same device as input."
        )
    return _bce_dispatch(input, target, weight, red, out=out)
