import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kunlunxin backend-local binary_cross_entropy_with_logits.
#
# Perf evidence (harness/solution/performance/binary_cross_entropy_with_logits_xpu3_20260817.md):
#  1. pointwise_dynamic 1D codegen uses a 512-lane tile: measured ~22 GB/s.
#     A single-tile-per-CTA copy at BLOCK=16384 reaches ~1.6 TB/s (torch
#     native copy: 1.85 TB/s).  Per-CTA dynamic `range` loops collapse to
#     ~4 GB/s (gm2lm cannot pipeline loads across dynamic loop iterations);
#     `tl.static_range` unrolling keeps the DMA path fast (~11 ms for the
#     full elementwise+reduce of 2^28 fp16, vs ~840 ms with the dynamic
#     loop).  So: static-unrolled kernels, BLOCK=16384, U=4-16.
#  2. The elementwise formula evaluates log(1+exp(-|x|)) once per element
#     (one exp + one log) instead of computing both branches; the other
#     branch is recovered exactly by identity log(1+e^x) = x + log(1+e^-x).
#     (2-exp + 2-log variant measured identical: the memory path, not the
#     transcendental count, is the wall on this backend.)
#  3. reduction=mean/sum no longer routes through the single-CTA mean kernel
#     (BLOCK up to 65536, one CTA -> tens-of-ms for 1e7+ elements).  A
#     two-stage fp32 split reduction (static-unrolled parallel partials +
#     tiny fold) is used and the pointwise tensor is never materialized.
# ---------------------------------------------------------------------------

# tl.sum safety cap on this backend is 8192 (no buffer) / 32768 (with
# buffer_size_limit=2048).  BLOCK=16384 + buffer_size_limit fulfils it.
_RED_BLOCK = 16384
_RED_U = 4
_FLAT_BLOCK = 2048
_FLAT_U = 8


def _flat_cfg(N):
    grid = max(1, triton.cdiv(N, _FLAT_BLOCK * _FLAT_U))
    return _FLAT_BLOCK, _FLAT_U, grid


def _need_mask(N, blk, u):
    return (N % (blk * u)) != 0


# ---------------------------- elementwise helpers ---------------------------

@triton.jit
def _bce_loss(xv, yv):
    # stable single-exp: log(1+e^-|x|) + max(x,0) - x*y
    return tl.log(1.0 + tl.exp(-tl.abs(xv))) + tl.maximum(xv, 0.0) - xv * yv


@triton.jit
def _bce_pos_weight_loss(xv, yv, pv):
    l = tl.log(1.0 + tl.exp(-tl.abs(xv)))
    neg_log = tl.where(xv >= 0, l, -xv + l)     # log(1+e^-x)
    pos_log = tl.where(xv >= 0, xv + l, l)      # log(1+e^x)
    x_pos = yv * pv * neg_log + (1.0 - yv) * (xv + neg_log)
    x_neg = yv * (-pv * xv + pv * pos_log) + (1.0 - yv) * pos_log
    return tl.where(xv >= 0.0, x_pos, x_neg)



# -------------------- fused reduction kernels (static-unroll) ---------------

@triton.jit
def _bce_reduce_kernel(
    x, y, mid, N, BLOCK: tl.constexpr, U: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
        acc += _bce_loss(xv, yv)
    tl.store(mid + pid, tl.sum(acc))


@triton.jit
def _bce_weight_reduce_kernel(
    x, y, w, mid, N, BLOCK: tl.constexpr, U: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + idx, mask=m, other=0.0).to(tl.float32)
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            wv = tl.load(w + idx).to(tl.float32)
        acc += _bce_loss(xv, yv) * wv
    tl.store(mid + pid, tl.sum(acc))


@triton.jit
def _bce_pos_weight_reduce_kernel(
    x, y, pw, mid, N, BLOCK: tl.constexpr, U: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            pv = tl.load(pw + idx, mask=m, other=0.0).to(tl.float32)
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            pv = tl.load(pw + idx).to(tl.float32)
        acc += _bce_pos_weight_loss(xv, yv, pv)
    tl.store(mid + pid, tl.sum(acc))


@triton.jit
def _bce_weight_pos_weight_reduce_kernel(
    x, y, w, pw, mid, N, BLOCK: tl.constexpr, U: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + idx, mask=m, other=0.0).to(tl.float32)
            pv = tl.load(pw + idx, mask=m, other=0.0).to(tl.float32)
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            wv = tl.load(w + idx).to(tl.float32)
            pv = tl.load(pw + idx).to(tl.float32)
        acc += _bce_pos_weight_loss(xv, yv, pv) * wv
    tl.store(mid + pid, tl.sum(acc))


# -------------------- flat pointwise kernels (reduction=0) -------------------
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
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
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
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(out + idx, (_bce_loss(xv, yv) * wv).to(out.dtype.element_ty), mask=m)
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            wv = tl.load(w + idx).to(tl.float32)
            tl.store(out + idx, (_bce_loss(xv, yv) * wv).to(out.dtype.element_ty))


@triton.jit
def _bce_pos_weight_flat_kernel(
    x, y, pw, out, N, BLOCK: tl.constexpr, U: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            pv = tl.load(pw + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(out + idx, _bce_pos_weight_loss(xv, yv, pv).to(out.dtype.element_ty), mask=m)
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            pv = tl.load(pw + idx).to(tl.float32)
            tl.store(out + idx, _bce_pos_weight_loss(xv, yv, pv).to(out.dtype.element_ty))


@triton.jit
def _bce_weight_pos_weight_flat_kernel(
    x, y, w, pw, out, N, BLOCK: tl.constexpr, U: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * U
    for i in tl.static_range(U):
        idx = base + i * BLOCK + tl.arange(0, BLOCK)
        if NEED_MASK:
            m = idx < N
            xv = tl.load(x + idx, mask=m, other=0.0).to(tl.float32)
            yv = tl.load(y + idx, mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w + idx, mask=m, other=0.0).to(tl.float32)
            pv = tl.load(pw + idx, mask=m, other=0.0).to(tl.float32)
            tl.store(
                out + idx,
                (_bce_pos_weight_loss(xv, yv, pv) * wv).to(out.dtype.element_ty),
                mask=m,
            )
        else:
            xv = tl.load(x + idx).to(tl.float32)
            yv = tl.load(y + idx).to(tl.float32)
            wv = tl.load(w + idx).to(tl.float32)
            pv = tl.load(pw + idx).to(tl.float32)
            tl.store(
                out + idx,
                (_bce_pos_weight_loss(xv, yv, pv) * wv).to(out.dtype.element_ty),
            )


# ------------------ pointwise_dynamic fallback (non-contiguous) -------------

@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_kernel(x, y):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    return tl.where(
        x_f32 >= 0,
        x_f32 - x_f32 * y_f32 + tl.log(1.0 + tl.exp(-x_f32)),
        tl.log(1.0 + tl.exp(x_f32)) - x_f32 * y_f32,
    )


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_weight_kernel(x, y, weight):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    w_f32 = weight.to(tl.float32)
    loss = tl.where(
        x_f32 >= 0,
        x_f32 - x_f32 * y_f32 + tl.log(1.0 + tl.exp(-x_f32)),
        tl.log(1.0 + tl.exp(x_f32)) - x_f32 * y_f32,
    )
    return loss * w_f32


@pointwise_dynamic(is_tensor=[True, True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _bce_pos_weight_kernel(x, y, pos_weight):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    pw_f32 = pos_weight.to(tl.float32)
    log_1p_exp_neg_x = tl.log(1.0 + tl.exp(-x_f32))
    log_1p_exp_x = tl.log(1.0 + tl.exp(x_f32))
    x_pos = y_f32 * pw_f32 * log_1p_exp_neg_x + (1.0 - y_f32) * (
        x_f32 + log_1p_exp_neg_x
    )
    x_neg = (
        y_f32 * (-pw_f32 * x_f32 + pw_f32 * log_1p_exp_x) + (1.0 - y_f32) * log_1p_exp_x
    )
    return tl.where(x_f32 >= 0, x_pos, x_neg)


@pointwise_dynamic(
    is_tensor=[True, True, True, True], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def _bce_weight_pos_weight_kernel(x, y, weight, pos_weight):
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    w_f32 = weight.to(tl.float32)
    pw_f32 = pos_weight.to(tl.float32)
    log_1p_exp_neg_x = tl.log(1.0 + tl.exp(-x_f32))
    log_1p_exp_x = tl.log(1.0 + tl.exp(x_f32))
    x_pos = y_f32 * pw_f32 * log_1p_exp_neg_x + (1.0 - y_f32) * (
        x_f32 + log_1p_exp_neg_x
    )
    x_neg = (
        y_f32 * (-pw_f32 * x_f32 + pw_f32 * log_1p_exp_x) + (1.0 - y_f32) * log_1p_exp_x
    )
    loss = tl.where(x_f32 >= 0, x_pos, x_neg)
    return loss * w_f32


# --------------------------------- wrapper ----------------------------------

# Small tensors go through the generic path: masked bulk tiles (other=0) are
# silently not honored on this backend for tiny N (verified in isolation), so
# the small region keeps the long-proven pointwise_dynamic + mean/sum path.
_SMALL_N = 16384
_BULK_BLOCK = 16384


def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=1
):
    logger.debug("GEMS_KUNLUNXIN BINARY_CROSS_ENTROPY_WITH_LOGITS")
    has_w = weight is not None
    has_pw = pos_weight is not None
    wargs = []
    if has_w:
        wargs.append(weight)
    if has_pw:
        wargs.append(pos_weight)

    use_flat = self.is_contiguous() and target.is_contiguous() and all(
        wgt.is_contiguous() for wgt in wargs
    )
    N = self.numel()

    if not use_flat or N == 0 or N <= _SMALL_N:
        # fallback: pointwise_dynamic handles arbitrary layouts/strides;
        # small tensors keep the proven generic path.
        if has_w and has_pw:
            out = _bce_weight_pos_weight_kernel(self, target, weight, pos_weight)
        elif has_w:
            out = _bce_weight_kernel(self, target, weight)
        elif has_pw:
            out = _bce_pos_weight_kernel(self, target, pos_weight)
        else:
            out = _bce_kernel(self, target)

        if reduction == 2:
            if N == 0:
                return torch.zeros((), dtype=self.dtype, device=self.device)
            flat = out.to(torch.float32).reshape(-1)
            chunk_size = 65536
            full_chunks = flat.numel() // chunk_size
            if full_chunks:
                total = (
                    flat[: full_chunks * chunk_size]
                    .reshape(full_chunks, chunk_size)
                    .sum(dim=1)
                    .sum()
                )
            else:
                total = torch.zeros((), dtype=torch.float32, device=flat.device)
            if full_chunks * chunk_size < flat.numel():
                total = total + flat[full_chunks * chunk_size :].sum()
            return total.to(self.dtype)
        if reduction == 1:
            if N == 0:
                return torch.full((), float("nan"), dtype=self.dtype, device=self.device)
            return out.mean()
        return out

    # ---- fast path: contiguous inputs ----
    if reduction == 0:
        blk, u, grid = _flat_cfg(N)
        need_mask = _need_mask(N, blk, u)
        out = torch.empty_like(self)
        with torch_device_fn.device(self.device):
            if has_w and has_pw:
                _bce_weight_pos_weight_flat_kernel[(grid,)](
                    self, target, weight, pos_weight, out, N,
                    BLOCK=blk, U=u, NEED_MASK=need_mask,
                )
            elif has_w:
                _bce_weight_flat_kernel[(grid,)](
                    self, target, weight, out, N, BLOCK=blk, U=u, NEED_MASK=need_mask,
                )
            elif has_pw:
                _bce_pos_weight_flat_kernel[(grid,)](
                    self, target, pos_weight, out, N, BLOCK=blk, U=u, NEED_MASK=need_mask,
                )
            else:
                _bce_flat_kernel[(grid,)](
                    self, target, out, N, BLOCK=blk, U=u, NEED_MASK=need_mask,
                )
        return out

    # mean (1) / sum (2): fused fp32 split reduction.
    # Masked loads (other=0) silently miscompile on this backend for several
    # tail geometries (device-wide status 700), so the reduce runs UNMASKED
    # over a zero-padded copy: pad elements (x=0, y=0, w=1, pw=0) contribute
    # exactly log(2) each, subtracted from the total afterwards.
    pad = (-N) % _BULK_BLOCK
    if pad:
        inp = torch.zeros(N + pad, dtype=self.dtype, device=self.device)
        targ = torch.zeros(N + pad, dtype=target.dtype, device=self.device)
        # gem copy_ override is slow on this backend; _copy_from is not
        # overridden and reaches the native copy engine.
        torch.ops.aten._copy_from(self.reshape(-1), inp[:N], False)
        torch.ops.aten._copy_from(target.reshape(-1), targ[:N], False)
        if has_w:
            rn = torch.ones(N + pad, dtype=weight.dtype, device=self.device)
            torch.ops.aten._copy_from(weight.reshape(-1), rn[:N], False)
        else:
            rn = None
        if has_pw:
            pwz = torch.zeros(N + pad, dtype=pos_weight.dtype, device=self.device)
            torch.ops.aten._copy_from(pos_weight.reshape(-1), pwz[:N], False)
        else:
            pwz = None
        total_n = N + pad
    else:
        inp, targ = self, target
        rn, pwz = weight, pos_weight
        total_n = N
    grid = (total_n // _BULK_BLOCK,)
    mid = torch.empty((grid[0],), dtype=torch.float32, device=self.device)
    with torch_device_fn.device(self.device):
        if has_w and has_pw:
            _bce_weight_pos_weight_reduce_kernel[grid](
                inp, targ, rn, pwz, mid, total_n,
                BLOCK=_BULK_BLOCK, U=1, NEED_MASK=False, buffer_size_limit=2048,
            )
        elif has_w:
            _bce_weight_reduce_kernel[grid](
                inp, targ, rn, mid, total_n,
                BLOCK=_BULK_BLOCK, U=1, NEED_MASK=False, buffer_size_limit=2048,
            )
        elif has_pw:
            _bce_pos_weight_reduce_kernel[grid](
                inp, targ, pwz, mid, total_n,
                BLOCK=_BULK_BLOCK, U=1, NEED_MASK=False, buffer_size_limit=2048,
            )
        else:
            _bce_reduce_kernel[grid](
                inp, targ, mid, total_n, BLOCK=_BULK_BLOCK, U=1, NEED_MASK=False,
                buffer_size_limit=2048,
            )

    total = mid.sum() - pad * 0.6931471805599453
    if reduction == 2:
        return total.to(self.dtype)
    return (total / N).to(self.dtype)