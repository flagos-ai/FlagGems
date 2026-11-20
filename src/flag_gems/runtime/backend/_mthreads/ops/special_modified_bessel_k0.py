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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.special_modified_bessel_k0 import (
    special_modified_bessel_k0 as default_special_modified_bessel_k0,
)
from flag_gems.ops.special_modified_bessel_k0 import (
    special_modified_bessel_k0_out as default_special_modified_bessel_k0_out,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only fp32 is specialized.
_SUPPORTED_DTYPES = {torch.float32}

_BLOCK = 1024
_MAX_BLOCK = 4096
_AUTO_MIN = 65536


@triton.jit
def _k0_body_f32(x):
    """K0(x) for float32: cephes/NR piecewise approximation."""
    # ---- small region: x <= 2 ----
    y = x * x * 0.25
    p = -0.57721566
    p = p + y * (
        0.42278420
        + y
        * (
            0.23069756
            + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))
        )
    )
    z = x * x * 0.07111111111111111  # (1/3.75)^2
    i0 = 1.0 + z * (
        3.5156229
        + z
        * (
            3.0899424
            + z * (1.2067492 + z * (0.2659732 + z * (0.0360768 + z * 0.0045813)))
        )
    )
    small = p - tl.log(x * 0.5) * i0

    # ---- large region: x > 2 ----
    t = 2.0 / x
    q = 1.25331414 + t * (
        -0.07832358
        + t
        * (
            0.02189568
            + t * (-0.01062446 + t * (0.00587872 + t * (-0.00251540 + t * 0.00053208)))
        )
    )
    large = q * tl.exp(-x) / tl.sqrt(x)

    return tl.where(x <= 2.0, small, large)


@triton.jit
def _k0_kernel_auto(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    xr = tl.load(x_ptr + offs)
    res = _k0_body_f32(xr.to(tl.float32))
    tl.store(out_ptr + offs, res.to(xr.dtype))


@triton.jit
def _k0_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    if EVEN:
        xr = tl.load(x_ptr + offs)
        res = _k0_body_f32(xr.to(tl.float32))
        tl.store(out_ptr + offs, res.to(xr.dtype))
    else:
        mask = offs < n_elements
        xr = tl.load(x_ptr + offs, mask=mask, other=1.0)
        res = _k0_body_f32(xr.to(tl.float32))
        tl.store(out_ptr + offs, res.to(xr.dtype), mask=mask)


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def special_modified_bessel_k0(self, out=None):
    logger.debug("GEMS_MTHREADS SPECIAL_MODIFIED_BESSEL_K0")

    x = self
    if not _use_triton_kernel(x):
        if out is None:
            return default_special_modified_bessel_k0(x)
        return default_special_modified_bessel_k0_out(x, out)

    if out is None:
        o = torch.empty_like(x)
    else:
        o = out
        if o.dtype != x.dtype:
            raise TypeError("out dtype must match input dtype")
        if o.device != x.device:
            raise TypeError("out device must match input device")

    xc = x.contiguous()
    oc = o.contiguous()

    n = xc.numel()
    if n > 0:
        with torch_device_fn.device(x.device):
            _launch_k0(x, xc, oc, n)
    if oc.data_ptr() != o.data_ptr():
        o.copy_(oc)
    return o


def _launch_k0(x, xc, oc, n):
    if n > 0:
        if n % _MAX_BLOCK == 0 and n >= _AUTO_MIN:
            # The whole-tile kernel needs no tail mask, so BLOCK may divide n
            # exactly; grid must be computed from that same BLOCK.
            grid = (triton.cdiv(n, _MAX_BLOCK),)
            _k0_kernel_auto[grid](xc, oc, n, BLOCK=_MAX_BLOCK)
        else:
            even = (n % _BLOCK) == 0
            grid = (triton.cdiv(n, _BLOCK),)
            _k0_kernel[grid](xc, oc, n, BLOCK=_BLOCK, EVEN=even)
