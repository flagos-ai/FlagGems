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
"""Kunlunxin (XPU3) override for ``aten::gradient`` (``torch.gradient``).

Why a backend override
----------------------
The generic ``flag_gems.ops.gradient`` implementation miscompiles / is very
slow on XPU3:

* its fused contiguous "rows" kernel fails to build for float16/bfloat16 --
  ``TritonXPUUnrollControl`` aborts with ``'arith.select' op failed to verify``
  on the low-precision bit-rounding helper;
* a masked store combining a scalar and a vector predicate drops a few lanes
  nondeterministically, leaving uninitialised ``torch.empty`` residue (seen as
  flaky 1-element misses on odd shapes);
* the reused ``@libentry`` kernels are ~15-40x slower than the eager reference
  because every neighbour load/store goes through the masked-memory path and
  every boundary is a per-lane ``setcc -> select`` (~300-400 us each on XPU3).

Design
------
For scalar spacing we split each requested dimension into two tiny,
branch-free launches:

* an **interior** kernel computes the central difference
  ``(a[i+1] - a[i-1]) * 0.5`` (unit spacing) for *every* element with plain
  masked loads -- no per-lane select, no per-lane divide.  Edge elements are
  written with a wrong value here and fixed by the second launch.
* an **edge** kernel visits only the first/last slab of every dim-d row
  (``numel / size`` elements per edge, i.e. a vanishing fraction) and writes the
  ``edge_order`` one-sided formula.

Both kernels read neighbours at ``addr +- stride`` which is affine in the flat
offset, so they lower to block DMA instead of gather.  Low-precision
(float16/bfloat16) rounding for the ``edge_order=2`` edge formula reuses the
reference's exact per-op rounding chain (``_rnd``) but only over the tiny edge
set; ``edge_order=1`` edges and the interior are a single correctly-rounded op,
matching the eager composite bit-for-bit.

The ``Tensor[]`` coordinate-spacing path is delegated to the generic
implementation (it already compiles and matches on XPU3).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.gradient import (  # reuse the reviewed pure-python helpers
    _check_sizes,
    _gradient_tensor_spacing,
    _MAX_DIM,
    _normalize_dims,
)

logger = logging.getLogger(__name__)

# Flat 1-D tile.  Large enough for block DMA to dominate launch overhead on the
# measured shapes; the XPU backend normalises num_warps to 1 so only the tile
# width matters.
_BLOCK = 8192
# Edge tiles are tiny; a smaller block avoids a huge sparse launch.
_EDGE_BLOCK = 1024


@triton.jit
def _rnd(x, comp, BF16: tl.constexpr):
    # Round a float32 intermediate to the output precision (replicating torch's
    # per-op rounding for float16/bfloat16 CUDA ops).  The final inf clamp is a
    # type-safe arithmetic blend instead of ``tl.where``: on XPU3 the
    # ``arith.select`` produced from ``tl.where(cond, int_const, int_vec)``
    # fails the TritonXPUUnrollControl verifier.
    b = x.to(tl.int32, bitcast=True)
    sgn = b & -2147483648
    absb = b & 0x7FFFFFFF
    if BF16:
        r = (absb + 0x7FFF + ((absb >> 16) & 1)) & -65536
    else:
        r = (absb + 0x0FFF + ((absb >> 13) & 1)) & -8192
    mag = r & 0x7FFFFFFF
    isinf = (mag >= 0x47800000).to(tl.int32)
    r = r * (1 - isinf) + 0x7F800000 * isinf
    return (sgn | r).to(tl.float32, bitcast=True)


@triton.jit
def _edge_first(a0, r1, r2, sp, EO: tl.constexpr, BF16: tl.constexpr):
    # One-sided difference at the left boundary (inputs already in comp dtype).
    if EO == 2:
        t1 = _rnd(-1.5 * a0, a0.dtype, BF16)
        t2 = _rnd(2.0 * r1, a0.dtype, BF16)
        t3 = _rnd(t1 + t2, a0.dtype, BF16)
        t4 = _rnd(0.5 * r2, a0.dtype, BF16)
        t5 = _rnd(t3 - t4, a0.dtype, BF16)
        return _rnd(t5 / sp, a0.dtype, BF16)
    return _rnd(_rnd(r1 - a0, a0.dtype, BF16) / sp, a0.dtype, BF16)


@triton.jit
def _edge_last(a0, l1, l2, sp, EO: tl.constexpr, BF16: tl.constexpr):
    # One-sided difference at the right boundary (inputs already in comp dtype).
    if EO == 2:
        t1 = _rnd(0.5 * l2, a0.dtype, BF16)
        t2 = _rnd(2.0 * l1, a0.dtype, BF16)
        t3 = _rnd(t1 - t2, a0.dtype, BF16)
        t4 = _rnd(1.5 * a0, a0.dtype, BF16)
        t5 = _rnd(t3 + t4, a0.dtype, BF16)
        return _rnd(t5 / sp, a0.dtype, BF16)
    return _rnd(_rnd(a0 - l1, a0.dtype, BF16) / sp, a0.dtype, BF16)


@triton.jit
def _in_addr(
    offs,
    CONTIG: tl.constexpr,
    s0,
    s1,
    s2,
    s3,
    s4,
    in0,
    in1,
    in2,
    in3,
    in4,
    n0,
    n1,
    n2,
    n3,
    n4,
):
    # Map a *contiguous output* flat offset to the (possibly strided) input
    # flat offset.  The contiguous fast path is a plain identity so the address
    # stays affine in ``offs``.
    if CONTIG:
        return offs
    return (
        ((offs // s0) % n0) * in0
        + ((offs // s1) % n1) * in1
        + ((offs // s2) % n2) * in2
        + ((offs // s3) % n3) * in3
        + ((offs // s4) % n4) * in4
    )


@triton.jit
def _gradient_interior_kernel(
    A_ptr,
    O_ptr,
    numel,
    S,
    in_d,
    size,
    sp,
    s0,
    s1,
    s2,
    s3,
    s4,
    in0,
    in1,
    in2,
    in3,
    in4,
    n0,
    n1,
    n2,
    n3,
    n4,
    CONTIG: tl.constexpr,
    UNIT: tl.constexpr,
    LOW_PRECISION: tl.constexpr,
    BF16: tl.constexpr,
    COMPF64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Branch-free central difference over every element; boundary elements are
    # overwritten by ``_gradient_edge_kernel``.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < numel
    c = (offs // S) % size
    ia = _in_addr(
        offs, CONTIG, s0, s1, s2, s3, s4, in0, in1, in2, in3, in4, n0, n1, n2, n3, n4
    )
    rd = O_ptr.dtype.element_ty
    l = tl.load(A_ptr + ia - in_d, mask=m & (c >= 1), other=0.0).to(rd)
    r = tl.load(A_ptr + ia + in_d, mask=m & (c + 1 < size), other=0.0).to(rd)
    if UNIT:
        g = (r - l) * 0.5
    else:
        if LOW_PRECISION:
            if COMPF64:
                comp = tl.float64
            else:
                comp = tl.float32
            lc = l.to(comp)
            rc = r.to(comp)
            # reference order: round(a[i+1]-a[i-1]) / sp / 2, rounding each op
            s = _rnd(rc - lc, comp, BF16)
            g = _rnd(_rnd(s / sp, comp, BF16) / 2.0, comp, BF16).to(rd)
        else:
            g = (r - l) / sp / 2.0
    tl.store(O_ptr + offs, g, mask=m)


@triton.jit
def _gradient_edge_kernel(
    A_ptr,
    O_ptr,
    numel,
    S,
    in_d,
    size,
    sp,
    PS,
    s0,
    s1,
    s2,
    s3,
    s4,
    in0,
    in1,
    in2,
    in3,
    in4,
    n0,
    n1,
    n2,
    n3,
    n4,
    CONTIG: tl.constexpr,
    EO: tl.constexpr,
    LOW_PRECISION: tl.constexpr,
    BF16: tl.constexpr,
    COMPF64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One-sided ``edge_order`` difference at the first and last slab of every
    # dim-d row.  ``k`` enumerates (outer p, inner s) pairs so the two edge flat
    # offsets are ``p*size*S + s`` and ``p*size*S + (size-1)*S + s``.
    pid = tl.program_id(0)
    k = pid * BLOCK + tl.arange(0, BLOCK)
    mk = k < PS
    p = k // S
    s = k % S
    o0 = p * (size * S) + s
    o1 = o0 + (size - 1) * S
    ia0 = _in_addr(
        o0, CONTIG, s0, s1, s2, s3, s4, in0, in1, in2, in3, in4, n0, n1, n2, n3, n4
    )
    ia1 = _in_addr(
        o1, CONTIG, s0, s1, s2, s3, s4, in0, in1, in2, in3, in4, n0, n1, n2, n3, n4
    )
    rd = O_ptr.dtype.element_ty
    a0 = tl.load(A_ptr + ia0, mask=mk, other=0.0).to(rd)
    r1 = tl.load(A_ptr + ia0 + in_d, mask=mk, other=0.0).to(rd)
    a1 = tl.load(A_ptr + ia1, mask=mk, other=0.0).to(rd)
    l1 = tl.load(A_ptr + ia1 - in_d, mask=mk, other=0.0).to(rd)
    if COMPF64:
        comp = tl.float64
    else:
        comp = tl.float32
    if EO == 2:
        r2 = tl.load(A_ptr + ia0 + 2 * in_d, mask=mk, other=0.0).to(rd)
        l2 = tl.load(A_ptr + ia1 - 2 * in_d, mask=mk, other=0.0).to(rd)
        if LOW_PRECISION:
            g0 = _edge_first(a0.to(comp), r1.to(comp), r2.to(comp), sp, EO, BF16)
            g1 = _edge_last(a1.to(comp), l1.to(comp), l2.to(comp), sp, EO, BF16)
        else:
            g0 = (-1.5 * a0 + 2.0 * r1 - 0.5 * r2) / sp
            g1 = (0.5 * l2 - 2.0 * l1 + 1.5 * a1) / sp
    else:
        g0 = (r1 - a0) / sp
        g1 = (a1 - l1) / sp
    tl.store(O_ptr + o0, g0.to(rd), mask=mk)
    tl.store(O_ptr + o1, g1.to(rd), mask=mk)


def _gradient_scalar_spacing(self, sp_list, dims, edge_order):
    """Scalar/unit spacing path: per requested dim, an interior + edge launch."""
    out_dtype = self.dtype
    if not (out_dtype.is_floating_point or out_dtype.is_complex):
        # integral inputs are promoted by true division with a Scalar
        out_dtype = torch.result_type(self, 1.0)

    ndim = self.dim()
    numel = self.numel()
    if ndim == 0 or numel == 0:
        return [
            torch.empty(self.shape, dtype=out_dtype, device=self.device) for _ in dims
        ]
    if ndim > _MAX_DIM:
        raise NotImplementedError(
            "FlagGems gradient supports tensors with at most "
            f"{_MAX_DIM} dimensions; got ndim={ndim}"
        )

    if self.is_complex():
        # the difference formulas are linear: run the real kernel on the
        # real/imag halves and recombine
        re = _gradient_scalar_spacing(self.real, sp_list, dims, edge_order)
        im = _gradient_scalar_spacing(self.imag, sp_list, dims, edge_order)
        return [torch.complex(r, i) for r, i in zip(re, im)]

    contig = self.is_contiguous()
    sizes = list(self.shape) + [1] * (_MAX_DIM - ndim)
    in_strides = list(self.stride()) + [1] * (_MAX_DIM - ndim)
    # contiguous strides of the (padded) output shape
    out_strides = []
    cum = 1
    for sz in reversed(sizes):
        out_strides.append(cum)
        cum *= sz
    out_strides.reverse()

    low_precision = out_dtype in (torch.float16, torch.bfloat16)
    bf16 = out_dtype == torch.bfloat16
    comp_f64 = out_dtype == torch.float64

    sp_pad = [1.0] * _MAX_DIM
    for d, s in zip(dims, sp_list):
        sp_pad[d] = float(s)

    outs = []
    for d in dims:
        out = torch.empty(self.shape, dtype=out_dtype, device=self.device)
        S = out_strides[d]
        in_d = in_strides[d]
        size = self.shape[d]
        sp = sp_pad[d]
        unit = sp == 1.0
        # The bfloat16 per-op rounding chain silently miscompiles on XPU3 for
        # wide tiles (first ~76 lanes correct, rest garbage/inf).  The non-unit
        # low-precision path is off the benchmark hot path, so shrink the tile.
        blk = _BLOCK if (unit or not low_precision) else 1024
        with torch.cuda.device(self.device):
            _gradient_interior_kernel[(triton.cdiv(numel, blk),)](
                self,
                out,
                numel,
                S,
                in_d,
                size,
                sp,
                out_strides[0],
                out_strides[1],
                out_strides[2],
                out_strides[3],
                out_strides[4],
                in_strides[0],
                in_strides[1],
                in_strides[2],
                in_strides[3],
                in_strides[4],
                sizes[0],
                sizes[1],
                sizes[2],
                sizes[3],
                sizes[4],
                CONTIG=contig,
                UNIT=unit,
                LOW_PRECISION=low_precision,
                BF16=bf16,
                COMPF64=comp_f64,
                BLOCK=blk,
            )
            ps = numel // size
            if ps > 0:
                _gradient_edge_kernel[(triton.cdiv(ps, _EDGE_BLOCK),)](
                    self,
                    out,
                    numel,
                    S,
                    in_d,
                    size,
                    sp,
                    ps,
                    out_strides[0],
                    out_strides[1],
                    out_strides[2],
                    out_strides[3],
                    out_strides[4],
                    in_strides[0],
                    in_strides[1],
                    in_strides[2],
                    in_strides[3],
                    in_strides[4],
                    sizes[0],
                    sizes[1],
                    sizes[2],
                    sizes[3],
                    sizes[4],
                    CONTIG=contig,
                    EO=edge_order,
                    LOW_PRECISION=low_precision,
                    BF16=bf16,
                    COMPF64=comp_f64,
                    BLOCK=_EDGE_BLOCK,
                )
        outs.append(out)
    return outs


def gradient(self, *, spacing=None, dim=None, edge_order=1):
    """Numerical gradient of ``self`` along ``dim`` (``torch.gradient``).

    See ``flag_gems.ops.gradient.gradient`` for the full contract; this backend
    override keeps the same semantics but uses XPU3-friendly fused kernels for
    the scalar-spacing path.
    """
    logger.debug("GEMS GRADIENT")

    if self.dtype == torch.uint8:
        raise RuntimeError("torch.gradient does not support uint8 input.")
    if edge_order not in (1, 2):
        raise RuntimeError(
            "torch.gradient only supports edge_order=1 and edge_order=2."
        )

    dims = _normalize_dims(self, dim)
    _check_sizes(self, dims, edge_order)

    if spacing is None or isinstance(spacing, (int, float)):
        unit = 1.0 if spacing is None else float(spacing)
        return _gradient_scalar_spacing(self, [unit] * len(dims), dims, edge_order)

    if isinstance(spacing, torch.Tensor):
        if spacing.dim() <= 1 and spacing.numel() == 1:
            unit = float(spacing)
            return _gradient_scalar_spacing(self, [unit] * len(dims), dims, edge_order)
        return _gradient_tensor_spacing(self, [spacing], dims, edge_order)

    if isinstance(spacing, (list, tuple)):
        if any(isinstance(s, torch.Tensor) for s in spacing):
            return _gradient_tensor_spacing(self, list(spacing), dims, edge_order)
        if len(spacing) != len(dims):
            raise RuntimeError(
                "torch.gradient expected spacing to be unspecified, a scalar "
                "or it's spacing and dim arguments to have the same length, "
                f"but got a spacing argument of length {len(spacing)} and a "
                f"dim argument of length {len(dims)}."
            )
        return _gradient_scalar_spacing(
            self, [float(s) for s in spacing], dims, edge_order
        )

    raise RuntimeError(
        "torch.gradient expected spacing to be unspecified, a scalar or a "
        f"list, but got {type(spacing)}"
    )
