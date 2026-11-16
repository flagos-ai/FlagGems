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

"""Kunlunxin XPU3 override for ``_weight_norm_differentiable_backward``.

Why this override exists
------------------------
The generic implementation builds a ``[BLOCK_GROUP, BLOCK_REDUCE]`` 2-D tile
and reduces it with ``tl.sum(products, axis=1)``.  On XPU3 axis-1 reductions
over a width that is not a multiple of 8 silently return wrong values (see
``KERNEL_OPT_EXPERIENCE.md`` rows 12/18/73), so every ``reduce_size % 8 != 0``
case fails (baseline: 9/27 correctness tests failed).  The generic kernel also
degenerates to per-lane strided memory ops, making it 10-100x slower than torch
on the large benchmark shapes.

Strategy (all tiles are 1-D or ``tl.dot``; there is no axis-1 ``tl.sum``)
------------------------------------------------------------------------
The operator is a contraction ``dot[g] = sum_r grad_w * saved_v`` followed by a
pointwise apply.  On this backend a wide 1-D ``tl.sum`` is the single most
expensive primitive (~10x the pointwise pass), and 1-D loop accumulators wider
than 64 lanes fail to compile (``uni_sram``).  The contraction is therefore
done with ``tl.dot`` against an all-ones matrix -- the documented XPU route for
cross-lane reductions -- with ``input_precision="ieee"`` so fp16/bf16 inputs
keep fp32 product accuracy (fp16 products lose ~3% on the dot, enough to blow
the grad_v tolerance).

The apply pass must broadcast a per-group ``scale``/``proj`` over a large
contiguous tile, which a naive 2-D tile does not do efficiently on XPU3.  Two
cheap 1-D forms are used instead:

* ``dim == last``: ``scale``/``proj`` are indexed by the flat offset modulo the
  row length (a mask when that length is a power of two), so the gather is
  affine and the whole pass is one flat tile.
* ``dim == 0``: ``scale``/``proj`` vary per row; a flat tile spanning a few rows
  rebuilds the broadcast with a handful of scalar selects (``tl.where``).

The host-side contract (contiguity checks, dim validation, dtype fusion gate,
higher-order-autograd decomposition) mirrors the generic implementation.  Only
``src/.../_kunlunxin/`` is touched.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.ops._weight_norm_differentiable_backward import _composite_backward
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# dim == 0 (reduce over contiguous rows)
# --------------------------------------------------------------------------- #
@libentry()
@triton.jit
def _wnb_dot_first_kernel(
    dotbuf,
    grad_g,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    M,
    N,
    R: tl.constexpr,
    C: tl.constexpr,
):
    """dot[i] = sum_j saved_v[i, j] * grad_w[i, j]; ``R`` rows per program."""
    pid = ext.program_id(0)
    r0 = pid * R
    rows = r0 + tl.arange(0, R)
    row_mask = rows < M
    rr = rows[:, None]
    ones = tl.full((C, 16), 1.0, dtype=tl.float32)
    acc = tl.zeros((R, 16), dtype=tl.float32)
    for c0 in range(0, N, C):
        cols = c0 + tl.arange(0, C)
        mask = row_mask[:, None] & (cols[None, :] < N)
        off = rr * N + cols[None, :]
        v = tl.load(saved_v + off, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(grad_w + off, mask=mask, other=0.0).to(tl.float32)
        acc += tl.dot(v * w, ones, out_dtype=tl.float32, input_precision="ieee")
    dot = tl.sum(acc, axis=1) * (1.0 / 16.0)

    g = tl.load(saved_g + rows, mask=row_mask, other=0.0).to(tl.float32)
    norm = tl.load(saved_norms + rows, mask=row_mask, other=1.0).to(tl.float32)
    tl.store(dotbuf + rows, dot, mask=row_mask)
    tl.store(grad_g + rows, dot / norm, mask=row_mask)


@libentry()
@triton.jit
def _wnb_apply_first_flat_kernel(
    grad_v,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    dotbuf,
    N: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Flat tile spanning ``ROWS`` rows; rebuild the per-row broadcast.

    ``ROWS`` and ``N`` are constexpr so the tile is exactly ``ROWS * N`` and the
    scalar selects fold; the caller guarantees ``M % ROWS == 0``.
    """
    pid = ext.program_id(0)
    r0 = pid * ROWS
    idx = tl.arange(0, ROWS * N)

    s0 = tl.load(saved_g + r0).to(tl.float32) / tl.load(saved_norms + r0).to(
        tl.float32
    )
    n0 = tl.load(saved_norms + r0).to(tl.float32)
    d0 = tl.load(dotbuf + r0).to(tl.float32)
    scale = tl.zeros((ROWS * N,), dtype=tl.float32) + s0
    proj = tl.zeros((ROWS * N,), dtype=tl.float32) + (d0 / (n0 * n0))
    for j in tl.static_range(1, ROWS):
        gj = tl.load(saved_g + r0 + j).to(tl.float32)
        nj = tl.load(saved_norms + r0 + j).to(tl.float32)
        dj = tl.load(dotbuf + r0 + j).to(tl.float32)
        cond = (idx >= j * N) & (idx < (j + 1) * N)
        scale = tl.where(cond, gj / nj, scale)
        proj = tl.where(cond, dj / (nj * nj), proj)

    base = r0 * N
    v = tl.load(saved_v + base + idx).to(tl.float32)
    w = tl.load(grad_w + base + idx).to(tl.float32)
    tl.store(grad_v + base + idx, scale * (w - v * proj))


@libentry()
@triton.jit
def _wnb_apply_first_kernel(
    grad_v,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    dotbuf,
    N,
    BLOCK: tl.constexpr,
):
    """General path: one row per program (handles any ``N``)."""
    idx = ext.program_id(0)
    base = idx * N
    g = tl.load(saved_g + idx).to(tl.float32)
    norm = tl.load(saved_norms + idx).to(tl.float32)
    dot = tl.load(dotbuf + idx).to(tl.float32)
    scale = g / norm
    proj = dot / (norm * norm)
    for c0 in range(0, N, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        mask = cols < N
        off = base + cols
        v = tl.load(saved_v + off, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(grad_w + off, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_v + off, scale * (w - v * proj), mask=mask)


# --------------------------------------------------------------------------- #
# dim == last (reduce over rows)
# --------------------------------------------------------------------------- #
@libentry()
@triton.jit
def _wnb_dot_last_kernel(
    dotbuf,
    grad_w,
    saved_v,
    M,
    N,
    R: tl.constexpr,
    C: tl.constexpr,
):
    """dot[j] = sum_i saved_v[i, j] * grad_w[i, j]; ``C`` columns per program.

    ``ones @ (v*w)`` reduces along the row axis with the matrix unit.  The 16
    identical result rows are written to ``dotbuf``; callers read row 0.
    """
    pid = ext.program_id(0)
    cols = pid * C + tl.arange(0, C)
    col_mask = cols < N
    ones = tl.full((16, R), 1.0, dtype=tl.float32)
    acc = tl.zeros((16, C), dtype=tl.float32)
    for r0 in range(0, M, R):
        rows = r0 + tl.arange(0, R)
        mask = (rows[:, None] < M) & col_mask[None, :]
        off = rows[:, None] * N + cols[None, :]
        v = tl.load(saved_v + off, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(grad_w + off, mask=mask, other=0.0).to(tl.float32)
        acc += tl.dot(ones, v * w, out_dtype=tl.float32, input_precision="ieee")
    tl.store(
        dotbuf + tl.arange(0, 16)[:, None] * N + cols[None, :],
        acc,
        mask=col_mask[None, :],
    )


@libentry()
@triton.jit
def _wnb_apply_last_flat_kernel(
    grad_v,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    dotbuf,
    TOT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flat pass; scale/proj are indexed by ``off % N`` (affine for pow-2 N)."""
    off = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < TOT
    grp = off % N
    dot = tl.load(dotbuf + grp, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(saved_g + grp, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(saved_norms + grp, mask=mask, other=1.0).to(tl.float32)
    scale = g / norm
    proj = dot / (norm * norm)
    v = tl.load(saved_v + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(grad_w + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(grad_v + off, scale * (w - v * proj), mask=mask)


@libentry()
@triton.jit
def _wnb_ggrad_last_kernel(
    grad_g,
    dotbuf,
    saved_norms,
    N,
    BLOCK: tl.constexpr,
):
    """grad_g when the flat apply cannot cover row 0 in one program."""
    cols = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    dot = tl.load(dotbuf + cols, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(saved_norms + cols, mask=mask, other=1.0).to(tl.float32)
    tl.store(grad_g + cols, dot / norm, mask=mask)


def _npo2(x):
    return max(1, triton.next_power_of_2(x))


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS _WEIGHT_NORM_DIFFERENTIABLE_BACKWARD")

    for name, tensor in (
        ("grad_w", grad_w),
        ("saved_v", saved_v),
        ("saved_g", saved_g),
        ("saved_norms", saved_norms),
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    if dim != 0 and dim != saved_v.ndim - 1:
        raise RuntimeError("Expected dim to be the first or last dimension")
    if saved_v.ndim == 0:
        raise IndexError("Dimension specified as -1 but tensor has no dimensions")

    broadcast_shape = [1] * saved_v.ndim
    broadcast_shape[dim] = saved_v.shape[dim]
    expected_norm_dtype = (
        torch.float32
        if saved_v.dtype in (torch.float16, torch.bfloat16)
        else saved_v.dtype
    )
    can_fuse = (
        not (
            torch.is_grad_enabled()
            and any(x.requires_grad for x in (grad_w, saved_v, saved_g, saved_norms))
        )
        and saved_v.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and grad_w.dtype == saved_v.dtype
        and saved_g.dtype == saved_v.dtype
        and saved_norms.dtype == expected_norm_dtype
        and grad_w.shape == saved_v.shape
        and list(saved_g.shape) == broadcast_shape
        and list(saved_norms.shape) == broadcast_shape
        and grad_w.device == saved_v.device == saved_g.device == saved_norms.device
        and saved_v.numel() > 0
    )
    if not can_fuse:
        # Higher-order autograd / unsupported dtype-layout: keep the generic
        # differentiable decomposition.  Semantic path, not a perf shortcut.
        return _composite_backward(grad_w, saved_v, saved_g, saved_norms, dim)

    grad_v = torch.empty_like(saved_v)
    grad_g = torch.empty_like(saved_g)
    if dim == 0:
        groups = saved_v.shape[0]
        reduce_size = math.prod(saved_v.shape[1:])
    else:
        groups = saved_v.shape[-1]
        reduce_size = math.prod(saved_v.shape[:-1])

    with torch_device_fn.device(saved_v.device):
        if dim == 0:
            r = 64
            c = max(16, min(64, _npo2(reduce_size)))
            dotbuf = torch.empty((groups,), dtype=torch.float32, device=saved_v.device)
            _wnb_dot_first_kernel[(triton.cdiv(groups, r),)](
                dotbuf,
                grad_g,
                grad_w,
                saved_v,
                saved_g,
                saved_norms,
                groups,
                reduce_size,
                R=r,
                C=c,
                num_warps=4,
            )
            rows = 4
            if (
                _is_pow2(reduce_size)
                and 256 <= reduce_size <= 8192
                and groups % rows == 0
            ):
                _wnb_apply_first_flat_kernel[(groups // rows,)](
                    grad_v,
                    grad_w,
                    saved_v,
                    saved_g,
                    saved_norms,
                    dotbuf,
                    N=reduce_size,
                    ROWS=rows,
                    num_warps=4,
                )
            else:
                block = max(16, min(1024, _npo2(reduce_size)))
                _wnb_apply_first_kernel[(groups,)](
                    grad_v,
                    grad_w,
                    saved_v,
                    saved_g,
                    saved_norms,
                    dotbuf,
                    reduce_size,
                    BLOCK=block,
                    num_warps=4,
                )
        else:
            r = max(16, min(64, _npo2(reduce_size)))
            c = max(16, min(256, _npo2(groups)))
            dotbuf = torch.empty(
                (16 * groups,), dtype=torch.float32, device=saved_v.device
            )
            _wnb_dot_last_kernel[(triton.cdiv(groups, c),)](
                dotbuf,
                grad_w,
                saved_v,
                reduce_size,
                groups,
                R=r,
                C=c,
                num_warps=4,
            )
            total = reduce_size * groups
            block = 8192 if total >= 65536 else 2048
            _wnb_apply_last_flat_kernel[(triton.cdiv(total, block),)](
                grad_v,
                grad_w,
                saved_v,
                saved_g,
                saved_norms,
                dotbuf,
                total,
                N=groups,
                BLOCK=block,
                num_warps=4,
            )
            _wnb_ggrad_last_kernel[(triton.cdiv(groups, 1024),)](
                grad_g,
                dotbuf,
                saved_norms,
                groups,
                BLOCK=1024,
                num_warps=4,
            )
    return grad_v, grad_g
