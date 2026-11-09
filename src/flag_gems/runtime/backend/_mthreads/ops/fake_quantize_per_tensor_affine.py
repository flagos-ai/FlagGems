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

"""Standalone Triton implementation of fake_quantize_per_tensor_affine.

Semantics (matching torch's per-tensor affine fake quantization):
    xf = float(x)
    q  = rint(xf / scale) + zero_point          # rint = round-half-to-even
    q  = clamp(q, quant_min, quant_max)
    y  = (q - zero_point) * scale               # cast back to x.dtype
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.fake_quantize_per_tensor_affine import (
    fake_quantize_per_tensor_affine as default_fake_quantize_per_tensor_affine,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so fp64 falls back to the
# generic implementation; fp16/bf16/fp32 are specialized.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit(do_not_specialize=["scale", "zero_point", "quant_min", "quant_max"])
def _fq_affine_1d(
    x_ptr,
    out_ptr,
    n_elements,
    scale,
    zero_point,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = scale.to(tl.float32)
    z = zero_point.to(tl.float32)
    lo = quant_min.to(tl.float32)
    hi = quant_max.to(tl.float32)

    y = x / s
    r = tl.floor(y)
    frac = y - r
    odd = (r - 2.0 * tl.floor(r * 0.5)) != 0.0
    q = r + tl.where((frac > 0.5) | ((frac == 0.5) & odd), 1.0, 0.0) + z
    q = tl.minimum(tl.maximum(q, lo), hi)
    out = (q - z) * s
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(
    do_not_specialize=[
        "n_elements",
        "d1",
        "d2",
        "d3",
        "p1",
        "p2",
        "st0",
        "st1",
        "st2",
        "st3",
        "scale",
        "zero_point",
        "quant_min",
        "quant_max",
    ]
)
def _fq_affine_strided(
    x_ptr,
    out_ptr,
    n_elements,
    d1,
    d2,
    d3,
    p1,
    p2,
    st0,
    st1,
    st2,
    st3,
    scale,
    zero_point,
    quant_min,
    quant_max,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements

    c0 = idx // p1
    rem = idx % p1
    c1 = rem // p2
    rem = rem % p2
    c2 = rem // d3
    c3 = rem % d3
    off = c0 * st0 + c1 * st1 + c2 * st2 + c3 * st3

    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    s = scale.to(tl.float32)
    z = zero_point.to(tl.float32)
    lo = quant_min.to(tl.float32)
    hi = quant_max.to(tl.float32)

    y = x / s
    r = tl.floor(y)
    frac = y - r
    odd = (r - 2.0 * tl.floor(r * 0.5)) != 0.0
    q = r + tl.where((frac > 0.5) | ((frac == 0.5) & odd), 1.0, 0.0) + z
    q = tl.minimum(tl.maximum(q, lo), hi)
    out = (q - z) * s
    tl.store(out_ptr + idx, out.to(out_ptr.dtype.element_ty), mask=mask)


_BLOCK = 1024
_BLOCK_SMALL = 512
_SPLIT = 1048576
_NUM_WARPS = 4


def _use_triton_kernel(input):
    """Whether ``input`` can be handled by the MUSA kernels."""
    if not isinstance(input, torch.Tensor):
        return False
    if input.device.type != "musa" or input.dtype not in _SUPPORTED_DTYPES:
        return False
    if input.is_complex():
        return False
    return True


def fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max):
    logger.debug("GEMS_MTHREADS FAKE_QUANTIZE_PER_TENSOR_AFFINE")

    if not _use_triton_kernel(input):
        return default_fake_quantize_per_tensor_affine(
            input, scale, zero_point, quant_min, quant_max
        )

    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    if isinstance(zero_point, torch.Tensor):
        zero_point = zero_point.item()
    zero_point = int(zero_point)
    quant_min = int(quant_min)
    quant_max = int(quant_max)

    if quant_min > quant_max:
        raise RuntimeError("quant_min must be less than or equal to quant_max")
    if zero_point < quant_min or zero_point > quant_max:
        raise RuntimeError("zero_point must be between quant_min and quant_max")

    scale = float(scale)
    n_elements = input.numel()
    out = torch.empty(input.shape, dtype=input.dtype, device=input.device)
    if n_elements == 0:
        return out

    with torch_device_fn.device(input.device):
        if input.is_contiguous():
            _fq_dispatch_contiguous(
                input, out, n_elements, scale, zero_point, quant_min, quant_max
            )
        else:
            _fq_dispatch_strided(
                input, out, n_elements, scale, zero_point, quant_min, quant_max
            )
    return out


def _fq_dispatch_contiguous(input, out, n_elements, scale, zero_point, quant_min, quant_max):
    # Sub-1M workloads are latency/parallelism bound rather than bandwidth
    # bound: a 512-element tile (4 elements per thread, 2x the CTAs of the
    # 1024 tile) measured 2.56-3.96us against 3.12-4.56us, while the 1M
    # inputs keep the 1024 tile.
    if n_elements < _SPLIT:
        _fq_affine_1d[(triton.cdiv(n_elements, _BLOCK_SMALL),)](
            input,
            out,
            n_elements,
            scale,
            zero_point,
            quant_min,
            quant_max,
            BLOCK=_BLOCK_SMALL,
            num_warps=_NUM_WARPS,
        )
    else:
        _fq_affine_1d[(triton.cdiv(n_elements, _BLOCK),)](
            input,
            out,
            n_elements,
            scale,
            zero_point,
            quant_min,
            quant_max,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )


def _fq_dispatch_strided(input, out, n_elements, scale, zero_point, quant_min, quant_max):
    # Generic non-contiguous path: decompose a flat logical index over rank<=4.
    shape = list(input.shape) + [1, 1, 1]
    stride = list(input.stride()) + [1, 1, 1]
    d1, d2, d3 = shape[1], shape[2], shape[3]
    p1 = d1 * d2 * d3
    p2 = d2 * d3
    _fq_affine_strided[(triton.cdiv(n_elements, _BLOCK),)](
        input,
        out,
        n_elements,
        d1,
        d2,
        d3,
        p1,
        p2,
        stride[0],
        stride[1],
        stride[2],
        stride[3],
        scale,
        zero_point,
        quant_min,
        quant_max,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
