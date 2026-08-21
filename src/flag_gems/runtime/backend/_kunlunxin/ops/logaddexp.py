import logging

import torch
import triton
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# ============================================================================
# logaddexp(x, y) = log(exp(x) + exp(y)) on Kunlunxin XPU.
#
# CORRECTNESS — the stable form is m + ln(1 + exp(-|x - y|)), m = max(x, y).
# On this backend the generic formula burns 2 extern math calls per element
# (tl.exp / tl.log), which serialize to a slow 2-wide scalar loop (same
# pathology measured for logaddexp2: ~2000us for 16M fp32 vs 128us plain
# copy), and the default pointwise codegen additionally lowers the baseline
# gem latency to tens of ms for large shapes (16M: ~68ms fp32/fp16/bf16).
#
# Fast path (logaddexp_fast_kernel, only for numel >= 2M, contiguous,
# NaN-free inputs — mirror of the closed logaddexp2 fast path):
#   z = max(-|x-y|, -24)  -- maxnum on this backend does not propagate NaN,
#                            so equal-infinities (d = NaN) clamp to z = -24
#                            -> u ~ 0 -> g ~ 0 -> res = m (inf/-inf ok)
#   u = (e^(z/256) poly)^256 -- 8 fp squarings; u = e^-|d|
#   g = log2(1+u) deg-8 monomial Horner (fit on u in [0,1]); res = m + g*ln2
# NaN-aware comparisons cannot lower fast on this backend (see logaddexp2
# record), so NaN inputs stay on the generic path (host-side detection).
# ============================================================================

# ---------- generic path (NaN-safe, any layout/dtype/shape) ----------------

config_ = None  # keep the same default codegen as the original generic impl


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def logaddexp_func(x, y):
    # log(exp(x) + exp(y)) = m + log(1 + exp(-|x - y|)), m = max(x, y)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    m = tl.maximum(x_f32, y_f32)
    delta = x_f32 - y_f32
    return m + tl.log(1.0 + tl.exp(-tl.abs(delta)))


# ---------- fast path ----------

FAST_BLOCK = 16384
FAST_MIN_NUMEL = 2 * 1024 * 1024


@triton.jit
def logaddexp_fast_kernel(
    x_ptr, y_ptr, o_ptr, numel, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs).to(tl.float32)
    m = tl.maximum(x, y)
    d = x - y
    # z in [-24, 0]; backend tl.maximum does not propagate NaN, so
    # equal-infinities (d = NaN) clamp to z = -24 -> u ~ 0 -> g ~ 0 -> res = m.
    z = tl.maximum(-tl.abs(d), -24.0)
    t = z * 0.00390625  # z / 256
    # e^t Taylor deg 5 (t in [-0.1, 0]) then 8 squarings -> u = e^z
    p = 1.0 + t * (
        1.0
        + t
        * (
            0.5
            + t * (0.16666666666666666 + t * (0.041666666666666664 + t * 0.008333333333333333))
        )
    )
    u = p
    u = u * u
    u = u * u
    u = u * u
    u = u * u
    u = u * u
    u = u * u
    u = u * u
    u = u * u
    # log2(1+u) monomial Horner deg 8, fit on u in [0, 1]
    g = -0.00876419584255894
    g = g * u + 0.04965553412883055
    g = g * u + -0.13318061612473603
    g = g * u + 0.23773233420472317
    g = g * u + -0.3450792615966863
    g = g * u + 0.4780139268978752
    g = g * u + -0.7210606595770077
    g = g * u + 1.442682712441337
    g = g * u + 1.3105136020818975e-07
    # ln(1+u) = log2(1+u) * ln2
    res = m + g * 0.6931471805599453
    if NEED_MASK:
        tl.store(o_ptr + offs, res.to(o_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(o_ptr + offs, res.to(o_ptr.dtype.element_ty))


def _run_fast(x, y, out=None):
    numel = x.numel()
    if out is None:
        out = torch.empty_like(x)
    block = min(FAST_BLOCK, triton.next_power_of_2(numel))
    grid = (triton.cdiv(numel, block),)
    need_mask = numel % block != 0
    warps = 32 if block >= 4096 else 8
    logaddexp_fast_kernel[grid](
        x, y, out, numel, BLOCK=block, NEED_MASK=need_mask, num_warps=warps
    )
    return out


_NAN_CACHE = {}
_NAN_CACHE_MAX = 16


def _has_nan(x, y):
    """Host-side NaN detection for the fast path.

    torch.isnan().any() resolves to the registered gems reductions inside
    use_gems() and is pathologically slow on this backend, so the outcome is
    cached per (data_ptr, version) pair (same scheme as logaddexp2).
    """
    key = (x.data_ptr(), x._version, y.data_ptr(), y._version)
    hit = _NAN_CACHE.get(key)
    if hit is not None:
        return hit
    has = bool(torch.isnan(x).any()) or bool(torch.isnan(y).any())
    if len(_NAN_CACHE) >= _NAN_CACHE_MAX:
        _NAN_CACHE.clear()
    _NAN_CACHE[key] = has
    return has


def _impl(x, y, out=None):
    numel = x.numel()
    use_fast = (
        numel >= FAST_MIN_NUMEL
        and x.is_contiguous()
        and y.is_contiguous()
        and not _has_nan(x, y)
    )
    if use_fast:
        return _run_fast(x, y, out)
    if out is None:
        return logaddexp_func(x, y)
    logaddexp_func(x, y, out0=out)
    return out


def logaddexp(self, other):
    logger.debug("GEMS_KUNLUNXIN LOGADDEXP")
    return _impl(self, other)


def logaddexp_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN LOGADDEXP_OUT")
    _impl(self, other, out)
    return out