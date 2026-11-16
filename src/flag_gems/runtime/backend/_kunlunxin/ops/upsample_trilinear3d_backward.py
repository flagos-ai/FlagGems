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
"""Kunlunxin (XPU3) override for ``aten::upsample_trilinear3d_backward``.

The upstream generic kernel scatters each output voxel into its eight input
corners with ``tl.atomic_add``.  On XPU3 that path is both numerically wrong in
the official test matrix (lost/duplicated updates -> O(1) absolute errors on
~0.2 % of the elements) and unsafe for the benchmark shapes (the benchmark
process aborts around the first large shape, consistent with the masked-atomic
out-of-bounds hazard that is documented for this backend).

This override computes the exact same result without atomics.  The trilinear
weight is separable,

    grad_in[nc, i, j, k] = sum_{o,p,q} grad_out[nc, o, p, q]
                           * wd(o, i) * wh(p, j) * ww(q, k)

so the backward is evaluated as three deterministic gather passes, one per
axis (W -> H -> D) over fp32 intermediates.  Each pass is a "reduce one axis"
kernel whose per-lane unrolled window is ``floor(2/scale) + 3`` positions
(<= 11 for the official matrix; 7 for the exact-2x benchmark matrix).  A single fused 3-D gather is *not* usable on
this backend: unrolling ``Kd*Kh*Kw`` (up to 120 loads) per lane overflows the
local-memory budget (``Failed to tune buffer size`` / ``uni_sram``), and a
dynamic nested loop trips ``TritonXPUUnrollControl``.

The per-axis weight is reproduced the way ATen computes it
(``area_pixel_compute_source_index`` + ``i0/i1`` lambda split, including the
``input_index0 == input_index1`` top-edge duplicate).  The weight formula was
verified against ``torch.ops.aten.upsample_trilinear3d_backward`` on CPU to
float32 round-off for align_corners in {True, False} and up/down/mixed scales.

All indices are int32; every gather address is clamped so that rejected
candidates and tail lanes always touch in-bounds memory (masked loads are not
honoured on this backend).  Outputs are over-allocated to a whole tile and tail
lanes recompute element 0 (harmless duplicate write).
"""

import logging
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_BLOCK = 1024


@libentry()
@triton.jit
def _axis_reduce_kernel(
    src,
    dst,
    total,
    IN_SIZE,
    OUT_SIZE,
    OUTER_STR_IN,
    AXIS_STR_IN,
    INNER,
    S,
    B,
    K: tl.constexpr,
    ALIGN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One lane per output element; gather over the window of one source axis.

    ``src`` is logically ``(outer, IN_SIZE, inner)`` and ``dst`` is
    ``(outer, OUT_SIZE, inner)``.  Both are contiguous, so the only strides
    needed are ``OUTER_STR_IN`` (outer stride of ``src``) and ``AXIS_STR_IN``.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    k = offsets % INNER
    t = offsets // INNER
    o = t % OUT_SIZE
    outer = t // OUT_SIZE

    # Tail lanes decode to garbage; force them onto element 0 so that a dropped
    # store mask only duplicates lane 0's value.
    k = tl.where(mask, k, 0)
    o = tl.where(mask, o, 0)
    outer = tl.where(mask, outer, 0)

    inv = tl.where(S == 0.0, 0.0, 1.0 / S)
    # ``- 1`` keeps a one-position margin on the low side: the analytic inverse
    # is exact in real arithmetic only, so an fp32 round-up must not clip the
    # lowest contributing source position.
    lo = tl.floor((o.to(tl.float32) - 1.0 - B) * inv).to(tl.int32) - 1
    lo = tl.maximum(lo, 0)

    base_in = outer * OUTER_STR_IN + k
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in tl.static_range(K):
        i = lo + dd
        ok = i < IN_SIZE
        ic = tl.minimum(i, IN_SIZE - 1)
        if ALIGN:
            s = ic.to(tl.float32) * S
        else:
            s = (ic.to(tl.float32) + 0.5) * S - 0.5
        s = tl.maximum(s, 0.0)
        s = tl.minimum(s, (OUT_SIZE - 1).to(tl.float32))
        i0 = s.to(tl.int32)
        i1 = tl.minimum(i0 + 1, OUT_SIZE - 1)
        lam = s - i0.to(tl.float32)
        w = tl.where(o == i0, 1.0 - lam, 0.0) + tl.where(o == i1, lam, 0.0)
        w = tl.where(ok, w, 0.0)
        value = tl.load(src + base_in + ic * AXIS_STR_IN)
        acc += value.to(tl.float32) * w

    tl.store(dst + offsets, acc.to(dst.dtype.element_ty), mask=mask)


def _scale_and_bias(in_size: int, out_size: int, align_corners: bool, scale):
    """Match ATen ``compute_scales_value<float>`` / ``compute_source_index``."""
    if align_corners:
        s = (float(in_size) - 1.0) / (float(out_size) - 1.0) if out_size > 1 else 0.0
        b = 0.0
    else:
        if scale is not None and scale > 0:
            s = 1.0 / float(scale)
        else:
            s = float(in_size) / float(out_size)
        b = 0.5 * s - 0.5
    return s, b


def _candidate_count(scale_units: float) -> int:
    """Contributing window per lane.

    The triangular basis is centred on the source coordinate with one unit of
    support on each side, so the contributing interval has length ``2/scale``
    in source-position units (at most ``floor(2/scale) + 1`` integers), plus one
    slot for the fp32 boundary slip and one for the ``- 1`` low margin.
    """
    if scale_units <= 0.0:
        return 1
    return int(2.0 / scale_units) + 3


def _axis_pass(
    src: torch.Tensor,
    outer: int,
    in_size: int,
    out_size: int,
    inner: int,
    out_dtype: torch.dtype,
    scale,
    align_corners: bool,
) -> torch.Tensor:
    """Reduce one axis of ``src`` (logical ``(outer, in_size, inner)``).

    ``in_size`` is the grad_output axis (source of the gather) and ``out_size``
    the grad_input axis (lane/destination axis, i.e. ATen's "input_size").
    """
    # ATen's source scale maps grad_output -> grad_input, so it is parameterised
    # as (input_size, output_size) = (grad_input, grad_output) here.
    s, b = _scale_and_bias(out_size, in_size, align_corners, scale)
    total = outer * out_size * inner
    buf = torch.empty(
        triton.cdiv(total, _BLOCK) * _BLOCK,
        device=src.device,
        dtype=out_dtype,
    )
    if total == 0:
        return buf[:0].view(outer, out_size, inner)
    dst = buf[:total]
    grid = (triton.cdiv(total, _BLOCK),)
    with torch_device_fn.device(src.device):
        _axis_reduce_kernel[grid](
            src,
            dst,
            total,
            in_size,
            out_size,
            in_size * inner,
            inner,
            inner,
            s,
            b,
            K=_candidate_count(s),
            ALIGN=bool(align_corners),
            BLOCK=_BLOCK,
            num_warps=1,
            buffer_size_limit=2048,
        )
    return buf[:total].view(outer, out_size, inner)


def upsample_trilinear3d_backward(
    grad_output: torch.Tensor,
    output_size: Sequence[int],
    input_size: Sequence[int],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_TRILINEAR3D_BACKWARD")
    assert grad_output.device.type == device.name
    assert grad_output.ndim == 5, "The ndim of grad_output must be 5"

    OD, OH, OW = (int(v) for v in output_size)
    N, C, ID, IH, IW = (int(v) for v in input_size)
    assert grad_output.shape == (N, C, OD, OH, OW), (
        f"grad_output shape {grad_output.shape} does not match "
        f"expected (N={N}, C={C}, OD={OD}, OH={OH}, OW={OW})"
    )

    nc = N * C
    grad_output = grad_output.contiguous()
    if nc * ID * IH * IW == 0 or nc * OD * OH * OW == 0:
        # No elements to reduce; an empty allocation is a valid result.
        return torch.empty(
            (N, C, ID, IH, IW), device=grad_output.device, dtype=grad_output.dtype
        )

    src = grad_output.view(nc, OD, OH, OW)
    # W: (nc*OD*OH, OW, 1) -> (nc*OD*OH, IW, 1)
    t1 = _axis_pass(
        src, nc * OD * OH, OW, IW, 1, torch.float32, scales_w, align_corners
    )
    # H: (nc*OD, OH, IW) -> (nc*OD, IH, IW)
    t2 = _axis_pass(
        t1, nc * OD, OH, IH, IW, torch.float32, scales_h, align_corners
    )
    # D: (nc, OD, IH*IW) -> (nc, ID, IH*IW)
    out = _axis_pass(
        t2, nc, OD, ID, IH * IW, grad_output.dtype, scales_d, align_corners
    )
    return out.view(N, C, ID, IH, IW)
