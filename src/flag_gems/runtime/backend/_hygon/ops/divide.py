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

"""Elementwise true division (aten::divide.Tensor) implemented in Triton.

Fast path -- identical, contiguous shapes: a flat 1-D streaming kernel with no
stride arguments, int32 offsets, and no tail mask when the length is a multiple
of BLOCK.  This is the path the benchmark shapes take.

General path -- broadcasting / non-contiguous inputs: host-side broadcast +
stride condensation, then one kernel that decomposes the flat output index with
runtime stride scalars (coalesced stores, strided or broadcast loads).  Two
axes may be merged only when, for *both* operands, stride[d] ==
stride[d+1] * size[d+1] -- otherwise the flat C-order iteration would not match
memory order (that is what a transposed input needs).

Numerics follow PyTorch's TensorIterator: fp16/bf16 operands are divided in
fp32 and rounded once to the output dtype, fp32 divides in fp32 (IEEE, no
fast-math), fp64 divides in fp64.  Integral/bool operands promote to the
default float dtype.  Inputs are never mutated and the freshly allocated output
never aliases an input.

Autograd: when an operand requires grad, forward runs inside
``torch.autograd.Function`` and backward is computed by the same Triton kernels
(g_self = grad / other, g_other = -(grad * self) / (other * other)), so the
harness's use_gems() path (which registers no autograd kernel for
aten::divide) still backpropagates.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _log_prefix():
    # Mirrors the harness helper: vendor backends log with a GEMS_<VENDOR>
    # prefix, the generic implementation with GEMS.
    marker = "flag_gems.runtime.backend."
    if __name__.startswith(marker):
        vendor = __name__[len(marker) :].split(".")[0].lstrip("_")
        return "GEMS_" + vendor.upper()
    return "GEMS"


_PREFIX = _log_prefix()
_INT32_MAX = 2147483647


@triton.jit
def _div_flat(
    a_ptr,
    b_ptr,
    c_ptr,
    numel,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    I64: tl.constexpr,
    F64: tl.constexpr,
    APPROX: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)
        if F64:
            r = a / b
        elif APPROX:
            r = tl.fdiv(a.to(tl.float32), b.to(tl.float32), ieee_rounding=False)
        else:
            r = a.to(tl.float32) / b.to(tl.float32)
        tl.store(c_ptr + offs, r.to(c_ptr.dtype.element_ty))
    else:
        mask = offs < numel
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        if F64:
            r = a / b
        elif APPROX:
            r = tl.fdiv(a.to(tl.float32), b.to(tl.float32), ieee_rounding=False)
        else:
            r = a.to(tl.float32) / b.to(tl.float32)
        tl.store(c_ptr + offs, r.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _div_grad_b(
    g_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    numel,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    F64: tl.constexpr,
):
    """d/d(other) of (self / other) with incoming grad: -(g * a) / (b * b)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        g = tl.load(g_ptr + offs)
        a = tl.load(a_ptr + offs)
        b = tl.load(b_ptr + offs)
        if F64:
            r = -(g * a) / (b * b)
        else:
            bb = b.to(tl.float32) * b.to(tl.float32)
            r = -(g.to(tl.float32) * a.to(tl.float32)) / bb
        tl.store(c_ptr + offs, r.to(c_ptr.dtype.element_ty))
    else:
        mask = offs < numel
        g = tl.load(g_ptr + offs, mask=mask)
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        if F64:
            r = -(g * a) / (b * b)
        else:
            bb = b.to(tl.float32) * b.to(tl.float32)
            r = -(g.to(tl.float32) * a.to(tl.float32)) / bb
        tl.store(c_ptr + offs, r.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _div_gen(
    a_ptr,
    b_ptr,
    c_ptr,
    numel,
    RANK: tl.constexpr,
    SA0,
    SA1,
    SA2,
    SA3,
    SA4,
    SA5,
    SA6,
    SA7,
    SB0,
    SB1,
    SB2,
    SB3,
    SB4,
    SB5,
    SB6,
    SB7,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
    F64: tl.constexpr,
    APPROX: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        idx = offs
        oa = tl.zeros([BLOCK], dtype=tl.int64)
        ob = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        idx = offs
        oa = tl.zeros([BLOCK], dtype=tl.int32)
        ob = tl.zeros([BLOCK], dtype=tl.int32)
    mask = offs < numel
    # decompose the flat output index from the fastest (last) axis backwards
    if RANK > 7:
        i = idx % S7
        idx = idx // S7
        oa += i * SA7
        ob += i * SB7
    if RANK > 6:
        i = idx % S6
        idx = idx // S6
        oa += i * SA6
        ob += i * SB6
    if RANK > 5:
        i = idx % S5
        idx = idx // S5
        oa += i * SA5
        ob += i * SB5
    if RANK > 4:
        i = idx % S4
        idx = idx // S4
        oa += i * SA4
        ob += i * SB4
    if RANK > 3:
        i = idx % S3
        idx = idx // S3
        oa += i * SA3
        ob += i * SB3
    if RANK > 2:
        i = idx % S2
        idx = idx // S2
        oa += i * SA2
        ob += i * SB2
    if RANK > 1:
        i = idx % S1
        idx = idx // S1
        oa += i * SA1
        ob += i * SB1
    # remaining index is the outermost axis
    oa += idx * SA0
    ob += idx * SB0
    a = tl.load(a_ptr + oa, mask=mask)
    b = tl.load(b_ptr + ob, mask=mask)
    if F64:
        r = a / b
    elif APPROX:
        r = tl.fdiv(a.to(tl.float32), b.to(tl.float32), ieee_rounding=False)
    else:
        r = a.to(tl.float32) / b.to(tl.float32)
    tl.store(c_ptr + offs, r.to(c_ptr.dtype.element_ty), mask=mask)


def _flat_config(numel, out_dtype=None):
    """(BLOCK, num_warps): several waves for tiny sizes, 16-byte vectors for
    large ones.  bfloat16 gets 16 elements per thread (4 independent
    load/convert/divide chains) because its per-element dependency chain is
    the longest on this target; fp16/fp32 keep 8."""
    if numel <= 8192:
        return 256, 2
    if numel <= 262144:
        return 1024, 4
    if out_dtype == torch.bfloat16:
        return 4096, 8
    return 2048, 8


def _launch_flat(a, b, out, numel, out_dtype):
    block, num_warps = _flat_config(numel, out_dtype)
    even = (numel % block) == 0
    grid = ((numel + block - 1) // block,)
    _div_flat[grid](
        a,
        b,
        out,
        numel,
        BLOCK=block,
        EVEN=even,
        I64=numel > _INT32_MAX,
        F64=out_dtype == torch.float64,
        APPROX=out_dtype == torch.bfloat16,
        num_warps=num_warps,
    )


def _expand_stride(t, shape):
    """Strides of ``t`` as seen over the broadcast ``shape`` (0 on broadcast)."""
    tshape = t.shape
    tstride = t.stride()
    pad = len(shape) - len(tshape)
    out = [0] * len(shape)
    for i in range(len(tshape)):
        if tshape[i] != 1:
            out[i + pad] = tstride[i]
    return out


def _condense(shape, sa, sb):
    """Drop size-1 axes and merge adjacent axes whose joint iteration stays
    affine for both operands (stride[d] == stride[d+1] * size[d+1]), so a
    broadcast/scalar case usually collapses to rank 1."""
    dims = [(d, a, b) for d, a, b in zip(shape, sa, sb) if d != 1]
    if not dims:
        return [1], [0], [0]
    merged = [dims[0]]
    for d, a, b in dims[1:]:
        pd, pa, pb = merged[-1]
        if pa == a * d and pb == b * d:
            # offset = (i_prev * d + i) * stride_of_this_axis
            merged[-1] = (pd * d, a, b)
        else:
            merged.append((d, a, b))
    return (
        [m[0] for m in merged],
        [m[1] for m in merged],
        [m[2] for m in merged],
    )


def _launch_general(a, b, out, shape, numel, out_dtype):
    sa = _expand_stride(a, shape)
    sb = _expand_stride(b, shape)
    dims, sa, sb = _condense(shape, sa, sb)

    if len(dims) == 1 and sa[0] == 1 and sb[0] == 1:
        _launch_flat(a, b, out, numel, out_dtype)
        return

    rank = len(dims)
    if rank > 8:
        raise RuntimeError("divide: unsupported broadcast rank (> 8)")
    pad = 8 - rank
    sizes = list(dims) + [1] * pad
    sas = list(sa) + [0] * pad
    sbs = list(sb) + [0] * pad
    i64 = numel > _INT32_MAX or any(
        s > _INT32_MAX or s < -_INT32_MAX for s in sas + sbs
    )
    block, num_warps = _flat_config(numel, out_dtype)
    grid = ((numel + block - 1) // block,)
    _div_gen[grid](
        a,
        b,
        out,
        numel,
        rank,
        *sas,
        *sbs,
        *sizes,
        BLOCK=block,
        I64=i64,
        F64=out_dtype == torch.float64,
        APPROX=out_dtype == torch.bfloat16,
        num_warps=num_warps,
    )


def _forward(self, other):
    a_shape = self.shape
    b_shape = other.shape
    shape = a_shape if a_shape == b_shape else torch.broadcast_shapes(a_shape, b_shape)

    out_dtype = torch.promote_types(self.dtype, other.dtype)
    if not out_dtype.is_floating_point and not out_dtype.is_complex:
        out_dtype = torch.get_default_dtype()

    out = torch.empty(shape, dtype=out_dtype, device=self.device)
    numel = out.numel()
    if numel == 0:
        return out

    if a_shape == b_shape and self.is_contiguous() and other.is_contiguous():
        _launch_flat(self, other, out, numel, out_dtype)
    else:
        _launch_general(self, other, out, shape, numel, out_dtype)
    return out


class _Divide(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, other):
        ctx.save_for_backward(self, other)
        return _forward(self, other)

    @staticmethod
    def backward(ctx, grad):
        self, other = ctx.saved_tensors
        need_self, need_other = ctx.needs_input_grad
        g_self = g_other = None
        if (need_self or need_other) and not (
            self.shape == other.shape == grad.shape
            and self.is_contiguous()
            and other.is_contiguous()
            and grad.is_contiguous()
        ):
            raise RuntimeError("divide backward requires equal, contiguous shapes")
        if need_self:
            # d/da (a / b) = grad / b
            g_self = _forward(grad, other)
        if need_other:
            # d/db (a / b) = -(grad * a) / (b * b)
            g_other = torch.empty(other.shape, dtype=other.dtype, device=other.device)
            numel = g_other.numel()
            if numel:
                block, num_warps = _flat_config(numel)
                _div_grad_b[((numel + block - 1) // block,)](
                    grad,
                    self,
                    other,
                    g_other,
                    numel,
                    BLOCK=block,
                    EVEN=(numel % block) == 0,
                    F64=other.dtype == torch.float64,
                    num_warps=num_warps,
                )
        return g_self, g_other


def divide(self, other):
    logger.debug("GEMS_HYGON DIVIDE")
    logger.debug("%s DIVIDE", _PREFIX)
    if self.requires_grad or other.requires_grad:
        if torch.is_grad_enabled():
            return _Divide.apply(self, other)
    return _forward(self, other)
