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

"""Adaptive average pooling 2D backward, implemented with a single-launch
Triton gather kernel.

Semantics (matching aten::_adaptive_avg_pool2d_backward(grad_output, self)):
  sh(oh) = (oh * H_in) // H_out,  eh(oh) = ((oh + 1) * H_in + H_out - 1) // H_out
  grad_input[nc, ih, iw] = sum_{oh, ow} [sh <= ih < eh] * [sw <= iw < ew]
                           * grad_output[nc, oh, ow] / ((eh-sh) * (ew-sw))

Inverting the window mapping gives, for every input pixel, the (small) range of
output rows/columns that contribute:
  oh in [ (ih * H_out) // H_in ,  ceil((ih + 1) * H_out / H_in) - 1 ]
so each input pixel is written exactly once (no zero-fill, no atomics) and the
number of gathered terms per pixel is at most ceil(H_out / H_in) + 1.

Two specializations:
  * integer-ratio (H_in % H_out == 0 and W_in % W_out == 0): every pixel has
    exactly one contributor, so the map collapses to one gather, one multiply
    and one coalesced store per element;
  * general: unrolled masked accumulation over the per-pixel contributor
    rectangle.

Geometry is passed as constexpr so the integer floor/ceil arithmetic is
strength-reduced, and BLOCK-wide stores stay fully coalesced.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _aap2d_backward_div_kernel(
    grad_output_ptr,
    grad_input_ptr,
    total,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    RH: tl.constexpr,
    RW: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Integer-ratio fast path (contiguous grad_output).

    With H_IN = RH * H_OUT and W_IN = RW * W_OUT, the flat input index
    `offs = nc * H_IN * W_IN + ih * W_IN + iw` maps to

        v  = offs // (W_IN * RH)   == nc * H_OUT + oh
        ow = (offs // RW) % W_OUT
        out_off = v * W_OUT + ow

    because ih = oh * RH + (ih % RH) and iw = ow * RW + (iw % RW).
    """
    pid = tl.program_id(0)
    if WIDE:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    v = offs // (W_IN * RH)
    ow = (offs // RW) % W_OUT
    g = tl.load(grad_output_ptr + (v * W_OUT + ow), mask=mask, other=0.0)
    tl.store(
        grad_input_ptr + offs,
        (g.to(ACC_TYPE) * SCALE).to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _aap2d_backward_vec_kernel(
    grad_output_ptr,
    grad_input_ptr,
    total,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    RH: tl.constexpr,
    RW: tl.constexpr,
    SCALE: tl.constexpr,
    V: tl.constexpr,
    B4: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Group-replicated integer-ratio fast path (contiguous grad_output).

    V consecutive input pixels (a power of two with V <= RW and V | RW) share
    one grad_output value, so one gather feeds V contiguous output elements.
    With the flat input index of a group start s0 = g * V,

        q = s0 // RW ;  r = q // W_OUT ;  ow = q - r * W_OUT
        v = r // RH  ;  out_off = v * W_OUT + ow

    is uniform over the group. This divides both the gather request count and
    the integer index chain by V while keeping one 1D-contiguous blocked
    layout: the load is a (B4,) vector, the broadcast goes along a trailing
    axis of length V and the store is a contiguous (B4, V) tile, so no
    shared-memory layout conversion is introduced.
    """
    pid = tl.program_id(0)
    g = pid * B4 + tl.arange(0, B4)
    s0 = g * V

    q = s0 // RW
    r = q // W_OUT
    ow = q - r * W_OUT
    v = r // RH
    src = v * W_OUT + ow
    if EVEN:
        val = tl.load(grad_output_ptr + src)
    else:
        val = tl.load(grad_output_ptr + src, mask=s0 < total, other=0.0)
    val = (val.to(ACC_TYPE) * SCALE).to(grad_input_ptr.dtype.element_ty)

    dst = s0[:, None] + tl.arange(0, V)[None, :]
    val2 = tl.broadcast_to(val[:, None], (B4, V))
    if EVEN:
        tl.store(
            grad_input_ptr + dst,
            val2,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )
    else:
        tl.store(
            grad_input_ptr + dst,
            val2,
            mask=dst < total,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )


@triton.jit
def _aap2d_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    total,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    C_SIZE: tl.constexpr,
    MAX_OH: tl.constexpr,
    MAX_OW: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    CONTIG: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Decompose the flat input index into (nc, ih, iw).
    t = offs // W_IN
    iw = offs - t * W_IN
    nc = t // H_IN
    ih = t - nc * H_IN

    # Inclusive range of output rows/columns whose window contains this pixel.
    oh0 = (ih * H_OUT) // H_IN
    oh_end = ((ih + 1) * H_OUT + (H_IN - 1)) // H_IN
    ow0 = (iw * W_OUT) // W_IN
    ow_end = ((iw + 1) * W_OUT + (W_IN - 1)) // W_IN

    if CONTIG:
        lead = nc.to(tl.int64) if WIDE else nc
        nc_off = (lead * H_OUT) * W_OUT
    else:
        n_idx = (nc // C_SIZE).to(tl.int64)
        c_idx = (nc - (nc // C_SIZE) * C_SIZE).to(tl.int64)
        nc_off = n_idx * stride_n + c_idx * stride_c

    acc = tl.zeros((BLOCK,), dtype=ACC_TYPE)

    for a in tl.static_range(MAX_OH):
        oh = oh0 + a
        ok_h = oh < oh_end
        sh = (oh * H_IN) // H_OUT
        eh = ((oh + 1) * H_IN + (H_OUT - 1)) // H_OUT
        kh = eh - sh
        for b in tl.static_range(MAX_OW):
            ow = ow0 + b
            sw_ = (ow * W_IN) // W_OUT
            ew = ((ow + 1) * W_IN + (W_OUT - 1)) // W_OUT
            area = (kh * (ew - sw_)).to(ACC_TYPE)
            m = mask & ok_h & (ow < ow_end)
            if CONTIG:
                if WIDE:
                    off = (nc.to(tl.int64) * H_OUT + oh.to(tl.int64)) * W_OUT + ow
                else:
                    off = nc_off + oh * W_OUT + ow
            else:
                off = nc_off + oh.to(tl.int64) * stride_h + ow.to(tl.int64) * stride_w
            g = tl.load(grad_output_ptr + off, mask=m, other=0.0)
            acc += g.to(ACC_TYPE) / area

    tl.store(grad_input_ptr + offs, acc.to(grad_input_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _zero_fill_kernel(out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        out_ptr + offs,
        tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty),
        mask=offs < total,
    )


_BLOCK = 1024
_VEC_GROUPS = 256
_NUM_WARPS = 4


def _adaptive_avg_pool2d_backward(grad_output, self, *args, **kwargs):
    logger.debug("GEMS_HYGON _ADAPTIVE_AVG_POOL2D_BACKWARD")
    ss = self.shape
    gs = grad_output.shape
    h_in = ss[-2]
    w_in = ss[-1]
    h_out = gs[-2]
    w_out = gs[-1]
    total = self.numel()

    grad_input = torch.empty(ss, dtype=self.dtype, device=self.device)

    if total == 0:
        return grad_input

    if h_out == 0 or w_out == 0:
        grid = ((total + _BLOCK - 1) // _BLOCK,)
        _zero_fill_kernel[grid](grad_input, total, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
        return grad_input

    acc_type = tl.float64 if self.dtype == torch.float64 else tl.float32
    wide = total > 2147483647
    grid = ((total + _BLOCK - 1) // _BLOCK,)

    rh = h_in // h_out
    rw = w_in // w_out
    contig = grad_output.is_contiguous()

    if contig and h_in == rh * h_out and w_in == rw * w_out:
        scale = 1.0 / (rh * rw)
        # Widest replication group the geometry allows: one group per thread
        # keeps the broadcast in-register. Sixteen-bit dtypes then fill one
        # 16-byte lane store per group; float32 needs two such stores but
        # still halves its gather and wavefront counts.
        v_cap = 8
        v_rep = 1
        while v_rep * 2 <= rw and v_rep * 2 <= v_cap:
            v_rep *= 2
        if not wide and v_rep > 1:
            v_block = _VEC_GROUPS * v_rep
            _aap2d_backward_vec_kernel[((total + v_block - 1) // v_block,)](
                grad_output,
                grad_input,
                total,
                H_IN=h_in,
                W_IN=w_in,
                H_OUT=h_out,
                W_OUT=w_out,
                RH=rh,
                RW=rw,
                SCALE=scale,
                V=v_rep,
                B4=_VEC_GROUPS,
                ACC_TYPE=acc_type,
                EVEN=total % v_block == 0,
                num_warps=_NUM_WARPS,
            )
            return grad_input
        _aap2d_backward_div_kernel[grid](
            grad_output,
            grad_input,
            total,
            H_IN=h_in,
            W_IN=w_in,
            H_OUT=h_out,
            W_OUT=w_out,
            RH=rh,
            RW=rw,
            SCALE=scale,
            BLOCK=_BLOCK,
            ACC_TYPE=acc_type,
            WIDE=wide,
            num_warps=_NUM_WARPS,
        )
        return grad_input

    ndim = grad_output.dim()
    c_size = gs[-3] if ndim >= 4 else 1
    _aap2d_backward_kernel[grid](
        grad_output,
        grad_input,
        total,
        grad_output.stride(-4) if ndim >= 4 else 0,
        grad_output.stride(-3),
        grad_output.stride(-2),
        grad_output.stride(-1),
        H_IN=h_in,
        W_IN=w_in,
        H_OUT=h_out,
        W_OUT=w_out,
        C_SIZE=c_size,
        MAX_OH=(h_out + h_in - 1) // h_in + 1,
        MAX_OW=(w_out + w_in - 1) // w_in + 1,
        BLOCK=_BLOCK,
        ACC_TYPE=acc_type,
        CONTIG=contig,
        WIDE=wide,
        num_warps=_NUM_WARPS,
    )
    return grad_input
