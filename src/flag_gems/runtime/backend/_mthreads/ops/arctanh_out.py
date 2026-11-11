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

"""FlagGems arctanh_out candidate (Triton, MUSA MTT S5000).

out[i] = atanh(self[i]) evaluated in fp32 opmath and written into the
caller-supplied ``out`` buffer, which is returned unchanged (identity kept).

Numerically exact form: the backend libdevice ``atanh``, evaluated in fp32
opmath for fp16/bf16 inputs and stored in the output dtype. This reproduces
PyTorch's CUDA/MUSA ``atanh`` bit-for-bit, including
  * |x| < 1  -> finite value (accurate for tiny |x|)
  * x == 1   -> +inf, x == -1 -> -inf
  * |x| > 1  -> NaN
  * x == NaN -> NaN
  * x == +-0 -> +-0 (sign preserved)
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _ld

from flag_gems.ops.arctanh import arctanh_out as default_arctanh_out
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only these dtypes are
# specialized; the rest fall back to the generic implementation.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_BLOCK = 1024
_WARPS = 4


@triton.jit
def _arctanh_contig(
    x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, USE_I64: tl.constexpr
):
    pid = tl.program_id(0)
    if USE_I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if x.dtype == tl.float64:
        y = _ld.atanh(x)
    else:
        xf = x.to(tl.float32)
        u = _ld.fast_dividef(1.0 + xf, 1.0 - xf)
        y = 0.34657359027997265 * tl.log2(u)
    tl.store(out_ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _arctanh_strided(
    x_ptr,
    out_ptr,
    n_elements,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    xs0,
    xs1,
    xs2,
    xs3,
    xs4,
    xs5,
    xs6,
    xs7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Slots are ordered last-dim-first; slots >= RANK are unused (shape 1).
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements

    rem = offs
    xoff = tl.zeros([BLOCK], dtype=tl.int64)
    ooff = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        i = rem % sh0
        rem = rem // sh0
        xoff += i * xs0
        ooff += i * os0
    if RANK > 1:
        i = rem % sh1
        rem = rem // sh1
        xoff += i * xs1
        ooff += i * os1
    if RANK > 2:
        i = rem % sh2
        rem = rem // sh2
        xoff += i * xs2
        ooff += i * os2
    if RANK > 3:
        i = rem % sh3
        rem = rem // sh3
        xoff += i * xs3
        ooff += i * os3
    if RANK > 4:
        i = rem % sh4
        rem = rem // sh4
        xoff += i * xs4
        ooff += i * os4
    if RANK > 5:
        i = rem % sh5
        rem = rem // sh5
        xoff += i * xs5
        ooff += i * os5
    if RANK > 6:
        i = rem % sh6
        rem = rem // sh6
        xoff += i * xs6
        ooff += i * os6
    if RANK > 7:
        i = rem % sh7
        rem = rem // sh7
        xoff += i * xs7
        ooff += i * os7

    x = tl.load(x_ptr + xoff, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    u = _ld.fast_dividef(1.0 + xf, 1.0 - xf)
    y = 0.34657359027997265 * tl.log2(u)
    tl.store(out_ptr + ooff, y.to(x.dtype), mask=mask)


def _strided_launch(x, out, n):
    shape = list(x.shape)[::-1]
    xstr = list(x.stride())[::-1]
    ostr = list(out.stride())[::-1]
    rank = len(shape)
    if rank == 0:
        shape, xstr, ostr, rank = [1], [0], [0], 1
    while rank > 8:
        # merge the two innermost dims when their union is a dense block
        merged = shape[0] * shape[1]
        if xstr[1] == shape[0] * xstr[0] and ostr[1] == shape[0] * ostr[0]:
            shape[0] = merged
            shape.pop(1)
            xstr.pop(1)
            ostr.pop(1)
            rank -= 1
        else:
            raise ValueError("arctanh_out: unsupported >8-D non-contiguous tensor")
    pads = 8 - rank
    shape = shape + [1] * pads
    xstr = xstr + [0] * pads
    ostr = ostr + [0] * pads
    grid = (triton.cdiv(n, _BLOCK),)
    _arctanh_strided[grid](
        x,
        out,
        n,
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
        shape[6],
        shape[7],
        xstr[0],
        xstr[1],
        xstr[2],
        xstr[3],
        xstr[4],
        xstr[5],
        xstr[6],
        xstr[7],
        ostr[0],
        ostr[1],
        ostr[2],
        ostr[3],
        ostr[4],
        ostr[5],
        ostr[6],
        ostr[7],
        RANK=rank,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )
    return out


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def arctanh_out(self=None, *args, out=None, **kwargs):
    logger.debug("GEMS_MTHREADS ARCTANH_OUT")

    x = self
    if x is None:
        for key in ("input", "x", "self"):
            if key in kwargs:
                x = kwargs.pop(key)
                break
    if x is None and args:
        x = args[0]
        args = args[1:]
    if out is None and args:
        out = args[0]
        args = args[1:]

    if not isinstance(x, torch.Tensor):
        raise TypeError("arctanh_out expects a Tensor input")

    if out is None:
        out = torch.empty_like(x)

    if not _use_triton_kernel(x):
        return default_arctanh_out(x, out)

    n = x.numel()
    if n == 0:
        return out

    with torch_device_fn.device(x.device):
        return _launch_arctanh_out(x, out, n)


def _launch_arctanh_out(x, out, n):
    if x.is_contiguous() and out.is_contiguous():
        use_i64 = n > 2147483647
        grid = (triton.cdiv(n, _BLOCK),)
        _arctanh_contig[grid](
            x,
            out,
            n,
            BLOCK=_BLOCK,
            USE_I64=use_i64,
            num_warps=_WARPS,
        )
        return out

    return _strided_launch(x, out, n)
