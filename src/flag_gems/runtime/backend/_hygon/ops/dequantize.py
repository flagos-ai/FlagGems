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

"""Triton implementation of aten::dequantize.self (per-tensor quantized tensors).

out[i] = (float32(code[i]) - float32(zero_point)) * float32(scale)

The quantized tensor's integer codes are obtained through a zero-copy dtype
view of its storage (no int_repr copy), so the kernel reads N bytes and writes
4N bytes, the minimum traffic for this pointwise op.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_BLOCK = 1024

# quantized dtype -> storage code dtype (same itemsize, so the view is exact)
_CODE_DTYPE = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}


@triton.jit
def _dequant_flat_kernel(
    q_ptr,
    out_ptr,
    n,
    scale,
    zp,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    q = tl.load(q_ptr + offs, mask=mask, other=0)
    v = (q.to(tl.float32) - zp) * scale
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def _dequant_strided_kernel(
    q_ptr,
    out_ptr,
    n,
    scale,
    zp,
    s0,
    s1,
    s2,
    s3,
    t0,
    t1,
    t2,
    t3,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Logical index offs -> physical storage offset, for non-contiguous input."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    rem = tl.where(mask, offs, 0)
    o = tl.zeros([BLOCK], dtype=tl.int32)
    if NDIM >= 1:
        o += (rem % s0) * t0
        rem = rem // s0
    if NDIM >= 2:
        o += (rem % s1) * t1
        rem = rem // s1
    if NDIM >= 3:
        o += (rem % s2) * t2
        rem = rem // s2
    if NDIM >= 4:
        o += (rem % s3) * t3
    q = tl.load(q_ptr + o, mask=mask, other=0)
    v = (q.to(tl.float32) - zp) * scale
    tl.store(out_ptr + offs, v, mask=mask)


def _canonical_layout(size, stride):
    """Drop unit dims and merge stride-compatible neighbours (metadata only)."""
    sizes = []
    strides = []
    for sz, st in zip(size, stride):
        if sz == 1:
            continue
        if sizes and strides[-1] == sz * st:
            sizes[-1] *= sz
        else:
            sizes.append(sz)
            strides.append(st)
    return sizes, strides


def dequantize(self, scale=None, zero_point=None):
    logger.debug("GEMS_HYGON DEQUANTIZE")
    dt = _CODE_DTYPE.get(self.dtype)
    if dt is not None:
        # Zero-copy reinterpretation of the quantized storage as plain integers.
        codes = self.view(dt)
        scale = self.q_scale()
        zero_point = self.q_zero_point()
    else:
        codes = self
        if scale is None:
            raise TypeError(
                "dequantize expects a quantized tensor, or (codes, scale[, zero_point])"
            )
        if zero_point is None:
            zero_point = 0

    n = codes.numel()
    contiguous = codes.is_contiguous()
    out = torch.empty(codes.shape, dtype=torch.float32, device=codes.device)
    if n == 0:
        return out

    block = _BLOCK
    grid = ((n + block - 1) // block,)
    if contiguous:
        _dequant_flat_kernel[grid](
            codes, out, n, float(scale), float(zero_point), block, False
        )
    else:
        sizes, strides = _canonical_layout(codes.size(), codes.stride())
        ndim = len(sizes)
        if ndim > 4:
            raise RuntimeError("unsupported layout: %d strided dims" % ndim)
        pad_s = sizes + [1] * (4 - ndim)
        pad_t = strides + [0] * (4 - ndim)
        _dequant_strided_kernel[grid](
            codes,
            out,
            n,
            float(scale),
            float(zero_point),
            pad_s[0],
            pad_s[1],
            pad_s[2],
            pad_s[3],
            pad_t[0],
            pad_t[1],
            pad_t[2],
            pad_t[3],
            ndim,
            block,
        )
    return out
