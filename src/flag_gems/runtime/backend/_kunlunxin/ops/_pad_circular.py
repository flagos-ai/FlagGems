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
"""Kunlunxin XPU3 override for ``aten::_pad_circular``.

The generic 1-D-pad kernel builds a *discrete* source vector
``sw = tl.where(sw < 0, sw + W, ...)`` and reads ``x[row * W + sw]``.  On this
backend that non-affine access is mis-lowered as a single contiguous DMA
anchored at the first lane, so the wrap corrections are dropped and the pad
columns read the wrong rows (12/30 numeric failures: every non-negative 1-D-pad
case of ``tests/test_pad_circular.py``).

This override keeps the same host contract but, for the non-negative 1-D-pad
case (``npad == 1``, ``pl >= 0``, ``pr >= 0``), splits each output row into at
most three fully **contiguous, unit-stride runs**:

    left  : out[0    : pl]     <- in[W - pl : W]
    body  : out[pl   : pl + W] <- in[0 : W]
    right : out[pl+W : Wo]     <- in[0 : pr]

Each run is a plain affine copy with a single prefix mask, the only access
shape this backend lowers to a contiguous DMA.  Addresses are clamped, never
computed from discrete vectors.  Every other case (negative crop, ``npad == 2``
or ``3``) delegates to the untouched generic implementation, which already
passes those cells.

The generic module function is rebound at import so the test's direct
``from flag_gems.ops._pad_circular import _pad_circular`` resolves here; the
vendor export also lands in ``flag_gems._pad_circular`` and therefore in the
``aten`` registration used by ``use_gems()``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Original generic entry, captured before the namespace is rebound.  Used to
# delegate the cases this override does not touch.
_ORIGINAL = None


@triton.jit
def _pad_circular_run_kernel(
    x_ptr,
    out_ptr,
    B,
    W,
    Wo,
    dst_off,
    src_off,
    run_len,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    wb = tl.program_id(1)
    # Clamp the block *start* (a scalar) so every lane address stays inside
    # [0, run_len) while ``cols`` keeps a unit-stride affine form.  Clamping the
    # vector with a runtime ``tl.minimum`` instead would turn this into a
    # discrete gather, which is mis-lowered (or ~100x slower) on this backend.
    start = wb * BLOCK
    start = tl.maximum(0, tl.minimum(start, run_len - BLOCK))
    cols = start + tl.arange(0, BLOCK)
    ok = cols < run_len
    src = x_ptr + r * W + src_off + cols
    dst = out_ptr + r * Wo + dst_off + cols
    vals = tl.load(src, mask=ok, other=0)
    tl.store(dst, vals, mask=ok)


def _launch_run(x, out, B, W, Wo, dst_off, src_off, run_len):
    block = min(max(1, triton.next_power_of_2(run_len)), 4096)
    grid = (B, triton.cdiv(run_len, block))
    _pad_circular_run_kernel[grid](
        x,
        out,
        B,
        W,
        Wo,
        dst_off,
        src_off,
        run_len,
        BLOCK=block,
    )


def _pad_circular_1d(x, pl, pr):
    in_shape = list(x.shape)
    W = in_shape[-1]
    Wo = W + pl + pr
    out_shape = in_shape[:-1] + [Wo]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    B = 1
    for s in in_shape[:-1]:
        B *= s
    if B == 0 or Wo == 0:
        return out

    with torch_device_fn.device(x.device):
        # body: out[pl : pl+W] <- in[0 : W]
        _launch_run(x, out, B, W, Wo, pl, 0, W)
        # left wrap: out[0 : pl] <- in[W-pl : W]
        if pl > 0:
            _launch_run(x, out, B, W, Wo, 0, W - pl, pl)
        # right wrap: out[pl+W : Wo] <- in[0 : pr]
        if pr > 0:
            _launch_run(x, out, B, W, Wo, pl + W, 0, pr)
    return out


def _pad_circular(x: torch.Tensor, pad):
    logger.debug("GEMS _PAD_CIRCULAR (kunlunxin)")

    ndim = x.dim()
    if len(pad) % 2 != 0:
        raise ValueError("padding length must be even")
    npad = len(pad) // 2
    if npad < 1 or npad > 3:
        raise ValueError(
            "flag_gems _pad_circular supports padding the last 1, 2 or 3 dimensions"
        )
    if ndim <= npad:
        raise ValueError("input must have at least one non-padded (batch) dimension")

    if npad == 1:
        pl = int(pad[0])
        pr = int(pad[1])
        W = x.shape[-1]
        if W + pl + pr <= 0:
            raise ValueError(
                "negative padding removed the whole dimension "
                f"{ndim - 1}: input {W} pad ({pl}, {pr})"
            )
        if pl > W or pr > W:
            raise ValueError("Padding value causes wrapping around more than once.")
        # Only the non-negative 1-D-pad case needs (and gets) the run-based
        # kernel; negative crop keeps the generic behaviour.
        if pl >= 0 and pr >= 0:
            return _pad_circular_1d(x.contiguous(), pl, pr)

    if _ORIGINAL is None:  # pragma: no cover - module import order guarantees this
        raise RuntimeError("generic _pad_circular unavailable for delegation")
    return _ORIGINAL(x, pad)


def _patch_generic_namespace():
    """Rebind ``flag_gems.ops._pad_circular._pad_circular`` to this override."""
    global _ORIGINAL
    import sys

    try:
        gen_mod = sys.modules.get("flag_gems.ops._pad_circular")
        if gen_mod is not None:
            if _ORIGINAL is None:
                _ORIGINAL = gen_mod._pad_circular
            gen_mod._pad_circular = _pad_circular
        gen_pkg = sys.modules.get("flag_gems.ops")
        if gen_pkg is not None and _ORIGINAL is not None:
            gen_pkg._pad_circular = _pad_circular
    except Exception:  # pragma: no cover - best effort rebinding
        pass


_patch_generic_namespace()


__all__ = ["_pad_circular"]
