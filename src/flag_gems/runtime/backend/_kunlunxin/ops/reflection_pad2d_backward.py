"""Kunlunxin (XPU) reflection_pad2d_backward.

The backward of reflection padding is a *separable* bilinear fold::

    gi[h, w] = sum_{oh in S_h} sum_{ow in S_w} go[oh, ow]

with |S_*| <= 3 (center + one reflection on each side).  Previous measured
implementations on this backend:

- torch-composite (narrow/add_/flip chains, ~15 launches): 2-22 ms.
- 2-pass separable fold (h-fold then w-fold, grid=(NC*H, cdiv(OW,BLOCK))):
  correct, but the program count is NC*H (e.g. 57344 programs for
  (16,64,56,56)); XPU per-program dispatch (~0.3us) dominates and the op
  measured 19-76 ms through the harness (0.014-0.019x of native).

This implementation is the single-pass *input-parallel* fold used by the
proven sibling reflection_pad3d_backward, specialized to 2D.  Each program
processes BLOCK contiguous input cells; every input cell accumulates the
(<=9) reflected grad_output contributions in fp32:

  - one shared row-base per (center, reflected) row select (3 of them: r0/rl/rr)
    and one shared column vector per (center, reflected) col select (3: c0/cl/cr)
    -> each load is a single r + c add, matching the pad3d structure;
  - reflected coordinates are clamped to their valid center value with
    tl.where, so every load is unconditional and in-bounds; per-contribution
    validity is reapplied with value-level tl.where(mask, v, 0.0) (guards the
    backend's known masked-load `other` quirk);
  - tail lanes (o >= total) are masked off the *store* (measured: a masked
    affine store costs the same as an unmasked one on this backend, whereas
    clamping the store index with tl.minimum breaks the affine store lowering
    and costs ~6x); the load decode stays on the clamped oc = min(o, total-1),
    so loads are never OOB either;
  - fp32 accumulation, single explicit store cast to the destination dtype.

XPU hazards respected (same rules as reflection_pad3d_backward):
- no tl.atomic_add (silently drops updates);
- no min/max on the vectorized *column* axis (defeats OffsetAnalysis); the
  reflected column is built with value-level tl.where(sel, reflected, center)
  selects whose both sides are in-bounds.

Note on perf (see also README): the op is six to ten times slower than the
native vendor kernel for the benchmark shapes (native runs at ~18 ps/cell,
~2 TB/s effective).  On this backend the input-parallel structure spends its
time in uncoalesced 9-load gathers plus program dispatch; every alternative
structured measured slower (2D tiles 10-100x worse, 2-pass 7-26x worse,
row-unrolled 2-pass 7-59 ms).  See harness/solution/reflection_pad2d_backward
for the full evidence chain.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _load_grad(grad_ptr, base, off, mask):
    """Load one reflected contribution; zero it when the contribution is off."""
    value = tl.load(grad_ptr + base + off).to(tl.float32)
    return tl.where(mask, value, 0.0)


@triton.jit
def _reflection_pad2d_backward_kernel(
    go_ptr,
    gi_ptr,
    total,
    OW,
    H,
    W,
    pad_t,
    pad_b,
    pad_l,
    pad_r,
    OUT_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """gi[oc] = sum of the (<=9) reflected grad_output cells for input cell oc.

    Grid: (cdiv(N*C*H*W, BLOCK),).  Loads use the clamped oc = min(o, total-1)
    so every address is in-bounds; the store is masked on o < total (a masked
    affine store is free here, a clamped store index is ~6x slower).
    """
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    oc = tl.minimum(o, total - 1)

    HW = H * W
    nc = oc // HW
    h = (oc // W) % H
    w = oc % W

    row_c = h + pad_t
    m_t = (h > 0) & (h <= pad_t)
    m_b = (h >= H - 1 - pad_b) & (h < H - 1)
    row_t = tl.where(m_t, pad_t - h, row_c)
    row_b = tl.where(m_b, pad_t + 2 * H - 2 - h, row_c)

    col_c = w + pad_l
    m_l = (w > 0) & (w <= pad_l)
    m_r = (w >= W - 1 - pad_r) & (w < W - 1)
    col_l = tl.where(m_l, pad_l - w, col_c)
    col_r = tl.where(m_r, pad_l + 2 * W - 2 - w, col_c)

    base = nc * (H + pad_t + pad_b) * OW
    r0 = base + row_c * OW
    rl = base + row_t * OW
    rr = base + row_b * OW

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    acc += tl.load(go_ptr + r0 + col_c).to(tl.float32)
    acc += _load_grad(go_ptr, r0, col_l, m_l)
    acc += _load_grad(go_ptr, r0, col_r, m_r)
    acc += _load_grad(go_ptr, rl, col_c, m_t)
    acc += _load_grad(go_ptr, rl, col_l, m_t & m_l)
    acc += _load_grad(go_ptr, rl, col_r, m_t & m_r)
    acc += _load_grad(go_ptr, rr, col_c, m_b)
    acc += _load_grad(go_ptr, rr, col_l, m_b & m_l)
    acc += _load_grad(go_ptr, rr, col_r, m_b & m_r)

    tl.store(gi_ptr + o, acc.to(OUT_DTYPE), mask=o < total)


def _normalize_padding(padding):
    if isinstance(padding, torch.Tensor):
        return tuple(int(p) for p in padding.tolist())
    if not isinstance(padding, (tuple, list)) or len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of 4 integers: "
            "(pad_left, pad_right, pad_top, pad_bottom)"
        )
    return tuple(int(p) for p in padding)


def reflection_pad2d_backward(grad_output, self, padding):
    """Backward of reflection_pad2d on Kunlunxin XPU.

    Args:
        grad_output: (N, C, H+pt+pb, W+pl+pr) gradient w.r.t. padded output.
        self: original (N, C, H, W) input (shape only).
        padding: [pad_left, pad_right, pad_top, pad_bottom].
    Returns:
        (N, C, H, W) gradient w.r.t. input.
    """
    logger.debug("GEMS_KUNLUNXIN REFLECTION_PAD2D_BACKWARD")

    pad_l, pad_r, pad_t, pad_b = _normalize_padding(padding)

    if self.dim() not in (3, 4):
        raise ValueError("input must be a 3D or 4D tensor")

    is_3d = self.dim() == 3
    x = self.contiguous()
    if is_3d:
        x = x.unsqueeze(0)
    go = grad_output.contiguous()
    if is_3d:
        go = go.unsqueeze(0)

    N, C, H, W = x.shape
    OH = H + pad_t + pad_b
    OW = W + pad_l + pad_r
    if tuple(go.shape) != (N, C, OH, OW):
        raise ValueError(
            f"grad_output shape {tuple(go.shape)} does not match "
            f"expected {(N, C, OH, OW)}"
        )

    if not (pad_l or pad_r or pad_t or pad_b):
        return grad_output.clone()

    NC = N * C
    total = NC * H * W
    if total == 0:
        return torch.empty(N, C, H, W, device=x.device, dtype=x.dtype)

    if x.dtype == torch.float16:
        out_dtype = tl.float16
    elif x.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float32

    gi = torch.empty(N, C, H, W, device=x.device, dtype=x.dtype)

    with torch_device_fn.device(x.device):
        BLOCK = 512
        _reflection_pad2d_backward_kernel[(triton.cdiv(total, BLOCK),)](
            go,
            gi,
            total,
            OW,
            H,
            W,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            OUT_DTYPE=out_dtype,
            BLOCK=BLOCK,
        )

    if is_3d:
        return gi.squeeze(0)
    return gi
