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

"""fake_quantize_per_tensor_affine_cachemask_backward (FlagGems `run` entry).

out = grad * mask with a boolean (0/1) mask, matching ATen bit-exactly:
  * mask != 0 -> out is grad bit-for-bit (NaN/Inf/-0.0 preserved),
  * mask == 0 -> out is IEEE `grad * 0`: signed zero following grad's sign and
    NaN when grad is non-finite.

On this target an arithmetic multiply, and a select between the literals
-0.0/0.0, both fold the sign of zero to +0.0 (verified numerically), which
breaks the operator contract.  The zero result is therefore built by masking
grad's sign bit and bitcasting back to float, which the compiler cannot fold.
Mask is loaded through its own pointer dtype (bool -> *i1 is bitcast to i8 by
the backend); no host-side dtype view or copy is needed.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 2048  # must match the BLOCK literal inlined in the kernels below
_NUM_WARPS = 8
_MAX_DIMS = 8


@triton.jit
def _fq_bwd_flat(grad_ptr, mask_ptr, out_ptr, n):
    offs = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    act = offs < n
    g = tl.load(grad_ptr + offs, mask=act, other=0.0)
    m = tl.load(mask_ptr + offs, mask=act, other=0)
    gf = g.to(tl.float32)
    ib = tl.cast(gf, tl.int32, bitcast=True)
    z = tl.cast(ib & -2147483648, tl.float32, bitcast=True)
    z = tl.where(tl.abs(gf) < float("inf"), z, float("nan"))
    y = tl.where(m != 0, gf, z)
    tl.store(out_ptr + offs, y.to(g.dtype), mask=act)


@triton.jit
def _fq_bwd_flat_exact(grad_ptr, mask_ptr, out_ptr):
    offs = tl.program_id(0) * 2048 + tl.arange(0, 2048)
    g = tl.load(grad_ptr + offs)
    m = tl.load(mask_ptr + offs)
    gf = g.to(tl.float32)
    ib = tl.cast(gf, tl.int32, bitcast=True)
    z = tl.cast(ib & -2147483648, tl.float32, bitcast=True)
    z = tl.where(tl.abs(gf) < float("inf"), z, float("nan"))
    y = tl.where(m != 0, gf, z)
    tl.store(out_ptr + offs, y.to(g.dtype))


@triton.jit
def _fq_bwd_strided(
    grad_ptr,
    mask_ptr,
    out_ptr,
    n,
    gs0,
    gs1,
    gs2,
    gs3,
    gs4,
    gs5,
    gs6,
    gs7,
    ms0,
    ms1,
    ms2,
    ms3,
    ms4,
    ms5,
    ms6,
    ms7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,
):
    offs = tl.program_id(0).to(tl.int64) * 2048 + tl.arange(0, 2048).to(tl.int64)
    act = offs < n
    rem = tl.where(act, offs, 0)
    goff = rem * 0
    moff = rem * 0
    ooff = rem * 0
    q = rem // d7
    r = rem - q * d7
    goff += r * gs7
    moff += r * ms7
    ooff += r * os7
    rem = q
    q = rem // d6
    r = rem - q * d6
    goff += r * gs6
    moff += r * ms6
    ooff += r * os6
    rem = q
    q = rem // d5
    r = rem - q * d5
    goff += r * gs5
    moff += r * ms5
    ooff += r * os5
    rem = q
    q = rem // d4
    r = rem - q * d4
    goff += r * gs4
    moff += r * ms4
    ooff += r * os4
    rem = q
    q = rem // d3
    r = rem - q * d3
    goff += r * gs3
    moff += r * ms3
    ooff += r * os3
    rem = q
    q = rem // d2
    r = rem - q * d2
    goff += r * gs2
    moff += r * ms2
    ooff += r * os2
    rem = q
    q = rem // d1
    r = rem - q * d1
    goff += r * gs1
    moff += r * ms1
    ooff += r * os1
    rem = q
    q = rem // d0
    r = rem - q * d0
    goff += r * gs0
    moff += r * ms0
    ooff += r * os0
    g = tl.load(grad_ptr + goff, mask=act, other=0.0)
    m = tl.load(mask_ptr + moff, mask=act, other=0)
    gf = g.to(tl.float32)
    ib = tl.cast(gf, tl.int32, bitcast=True)
    z = tl.cast(ib & -2147483648, tl.float32, bitcast=True)
    z = tl.where(tl.abs(gf) < float("inf"), z, float("nan"))
    y = tl.where(m != 0, gf, z)
    tl.store(out_ptr + ooff, y.to(g.dtype), mask=act)


def _pad_stride(seq):
    pad = _MAX_DIMS - len(seq)
    return [0] * pad + list(seq)


def _pad_shape(seq):
    pad = _MAX_DIMS - len(seq)
    return [1] * pad + list(seq)


def fake_quantize_per_tensor_affine_cachemask_backward(grad, mask, *args, **kwargs):
    out = torch.empty_like(grad)
    n = grad.numel()
    if n == 0:
        return out
    if grad.is_contiguous() and mask.is_contiguous():
        if n & (_BLOCK - 1) == 0:
            _fq_bwd_flat_exact[(n // _BLOCK,)](grad, mask, out, num_warps=_NUM_WARPS)
        else:
            grid = ((n + _BLOCK - 1) // _BLOCK,)
            _fq_bwd_flat[grid](grad, mask, out, n, num_warps=_NUM_WARPS)
        return out
    if grad.dim() > _MAX_DIMS:
        raise NotImplementedError("non-contiguous input with rank > 8")
    gs = _pad_stride(grad.stride())
    ms = _pad_stride(mask.stride())
    os_ = _pad_stride(out.stride())
    ds = _pad_shape(grad.shape)
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    _fq_bwd_strided[grid](
        grad,
        mask,
        out,
        n,
        gs[0],
        gs[1],
        gs[2],
        gs[3],
        gs[4],
        gs[5],
        gs[6],
        gs[7],
        ms[0],
        ms[1],
        ms[2],
        ms[3],
        ms[4],
        ms[5],
        ms[6],
        ms[7],
        os_[0],
        os_[1],
        os_[2],
        os_[3],
        os_[4],
        os_[5],
        os_[6],
        os_[7],
        ds[0],
        ds[1],
        ds[2],
        ds[3],
        ds[4],
        ds[5],
        ds[6],
        ds[7],
        num_warps=_NUM_WARPS,
    )
    return out
