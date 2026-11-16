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

"""Fake-quantize per-tensor affine with exact in-range cachemask.

One fused Triton kernel reads the input once and writes both the dequantized
tensor and the boolean cachemask.  All arithmetic is fp32 (what ATen uses
internally): fp32 reciprocal of the fp32-rounded scale, round-half-to-even,
then a clamp.  Semantics were aligned against the target's ATen reference.
"""

import logging
import math
import struct

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _ld

from flag_gems.ops.fake_quantize_per_tensor_affine_cachemask import (
    fake_quantize_per_tensor_affine_cachemask as default_fq_cachemask,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only these dtypes are
# specialized; the rest fall back to the generic implementation.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# MUSA libdevice implements rint (round-half-to-even); nearbyint raises
# NotImplementedError and round() is round-half-away, so only rint is usable.

# 2**63: ATen converts the rounded value to int64, so values at or beyond this
# magnitude overflow.  The target's conversion sends positive overflow to the
# int minimum (clamped to quant_min) and negative overflow to the quant_max
# side, which also covers +inf / -inf.
_P63 = tl.constexpr(9223372036854775808.0)
_N63 = tl.constexpr(-9223372036854775808.0)

_BLOCK = 1024


@triton.jit
def _quant(x, scale, inv_scale, zero_point, quant_min, quant_max):
    """ATen-equivalent fake quant for one fp32 value; returns (result, keep)."""
    u = _ld.rint(x * inv_scale) + zero_point
    isnan = u != u
    keep = ((u >= quant_min) & (u <= quant_max)) | (
        isnan & (quant_min <= 0.0) & (0.0 <= quant_max)
    )
    # ATen's int64 conversion maps NaN to 0 and sends out-of-range magnitudes
    # (including +-inf) past the quantization range in a fixed direction.
    u = tl.where(
        isnan,
        0.0,
        tl.where(u >= _P63, quant_min, tl.where(u < _N63, quant_max, u)),
    )
    q = tl.maximum(tl.minimum(u, quant_max), quant_min)
    return (q - zero_point) * scale, keep


@triton.jit(
    do_not_specialize=["scale", "inv_scale", "zero_point", "quant_min", "quant_max"]
)
def _fq_flat_kernel(
    in_ptr,
    out_ptr,
    mask_ptr,
    n_elem,
    scale,
    inv_scale,
    zero_point,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offs < n_elem
    x = tl.load(in_ptr + offs, mask=active, other=0.0).to(tl.float32)
    res, keep = _quant(x, scale, inv_scale, zero_point, quant_min, quant_max)
    tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=active)
    tl.store(mask_ptr + offs, keep, mask=active)


@triton.jit(
    do_not_specialize=["scale", "inv_scale", "zero_point", "quant_min", "quant_max"]
)
def _fq_strided_kernel(
    in_ptr,
    out_ptr,
    mask_ptr,
    meta_ptr,
    n_elem,
    scale,
    inv_scale,
    zero_point,
    quant_min,
    quant_max,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    active = offs < n_elem
    in_off = tl.zeros((BLOCK,), tl.int64)
    out_off = tl.zeros((BLOCK,), tl.int64)
    msk_off = tl.zeros((BLOCK,), tl.int64)
    for k in tl.static_range(RANK):
        base = k * 5
        size = tl.load(meta_ptr + base + 0)
        div = tl.load(meta_ptr + base + 1)
        idx = (offs // div) % size
        in_off += idx * tl.load(meta_ptr + base + 2)
        out_off += idx * tl.load(meta_ptr + base + 3)
        msk_off += idx * tl.load(meta_ptr + base + 4)
    x = tl.load(in_ptr + in_off, mask=active, other=0.0).to(tl.float32)
    res, keep = _quant(x, scale, inv_scale, zero_point, quant_min, quant_max)
    tl.store(out_ptr + out_off, res.to(out_ptr.dtype.element_ty), mask=active)
    tl.store(mask_ptr + msk_off, keep, mask=active)


def _f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


def _decompose(inp, out, mask):
    shape = inp.shape
    rank = len(shape)
    divs = [1] * rank
    for k in range(rank - 2, -1, -1):
        divs[k] = divs[k + 1] * shape[k + 1]
    istr = inp.stride()
    ostr = out.stride()
    mstr = mask.stride()
    dims = [
        (shape[k], divs[k], istr[k], ostr[k], mstr[k])
        for k in range(rank)
        if shape[k] != 1
    ]
    if not dims:
        dims.append((1, 1, 0, 0, 0))
    return dims


def _launch(inp, out, mask, scale_f, inv_scale, zp_f, qmin_f, qmax_f):
    n = inp.numel()
    grid = ((n + _BLOCK - 1) // _BLOCK,)
    if inp.is_contiguous() and out.is_contiguous() and mask.is_contiguous():
        _fq_flat_kernel[grid](
            inp,
            out,
            mask,
            n,
            scale_f,
            inv_scale,
            zp_f,
            qmin_f,
            qmax_f,
            BLOCK=_BLOCK,
        )
        return
    dims = _decompose(inp, out, mask)
    meta = torch.tensor(dims, dtype=torch.int64, device=inp.device)
    _fq_strided_kernel[grid](
        inp,
        out,
        mask,
        meta,
        n,
        scale_f,
        inv_scale,
        zp_f,
        qmin_f,
        qmax_f,
        RANK=len(dims),
        BLOCK=_BLOCK,
    )


def _parse(self, scale, zero_point, quant_min, quant_max, args, kwargs):
    out0 = out1 = None
    extra = []
    for a in args:
        if isinstance(a, torch.Tensor):
            extra.append(a)
        elif isinstance(a, (tuple, list)) and len(a) == 2:
            out0, out1 = a
        elif isinstance(a, (tuple, list)) and len(a) == 1:
            out0 = a[0]
    if len(extra) == 1:
        out0 = extra[0]
    elif len(extra) >= 2:
        out0, out1 = extra[0], extra[1]

    for key, val in kwargs.items():
        if key in ("self", "input", "x", "inp", "input_tensor"):
            self = val
        elif key in ("scale", "sc"):
            scale = val
        elif key in ("zero_point", "zp"):
            zero_point = val
        elif key in ("quant_min", "qmin"):
            quant_min = val
        elif key in ("quant_max", "qmax"):
            quant_max = val
        elif key in ("out", "out0", "output", "out_output"):
            if key == "out" and isinstance(val, (tuple, list)):
                out0, out1 = val[0], val[1]
            else:
                out0 = val
        elif key in ("out1", "mask", "out_mask"):
            out1 = val
    return self, scale, zero_point, quant_min, quant_max, out0, out1


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def fake_quantize_per_tensor_affine_cachemask(
    self=None,
    scale=None,
    zero_point=None,
    quant_min=None,
    quant_max=None,
    *args,
    **kwargs,
):
    logger.debug("GEMS_MTHREADS FAKE_QUANTIZE_PER_TENSOR_AFFINE_CACHEMASK")

    if not _use_triton_kernel(self):
        return default_fq_cachemask(
            self, scale, zero_point, quant_min, quant_max, *args, **kwargs
        )

    if args or kwargs:
        self, scale, zero_point, quant_min, quant_max, out0, out1 = _parse(
            self, scale, zero_point, quant_min, quant_max, args, kwargs
        )
    else:
        out0 = out1 = None

    sf = float(scale)
    zp = int(zero_point)
    qmin = int(quant_min)
    qmax = int(quant_max)

    if qmin > qmax:
        raise RuntimeError(
            "`quant_min` should be less than or         equal to `quant_max`."
        )
    if zp < qmin or zp > qmax:
        raise RuntimeError("`zero_point` must be between `quant_min` and `quant_max`.")

    if out0 is not None:
        if out0.dtype != self.dtype:
            raise RuntimeError(
                f"Expected out tensor to have dtype {self.dtype}, "
                f"but got {out0.dtype} instead"
            )
        if out0.device != self.device:
            raise RuntimeError("input and out tensors must be on the same device")
    if out1 is not None:
        if out1.dtype != torch.bool:
            raise RuntimeError(
                f"Expected out tensor to have dtype torch.bool, "
                f"but got {out1.dtype} instead"
            )
        if out1.device != self.device:
            raise RuntimeError("input and out tensors must be on the same device")

    shape = self.shape
    if out0 is None:
        out0 = torch.empty_like(self)
    elif out0.shape != shape:
        out0.resize_(shape)
    if out1 is None:
        out1 = torch.empty_like(self, dtype=torch.bool)
    elif out1.shape != shape:
        out1.resize_(shape)

    if self.numel() == 0:
        return out0, out1

    scale_f = _f32(sf)
    if scale_f == 0.0:
        inv_scale = math.copysign(math.inf, scale_f)
    else:
        inv_scale = _f32(1.0 / scale_f)
    with torch_device_fn.device(self.device):
        _launch(
            self,
            out0,
            out1,
            scale_f,
            inv_scale,
            float(zp),
            float(qmin),
            float(qmax),
        )
    return out0, out1
