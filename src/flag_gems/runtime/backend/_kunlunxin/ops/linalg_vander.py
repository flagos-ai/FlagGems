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

from .vander import _BLOCK_COMPLEX
from .vander import _BLOCK_FLOAT
from .vander import _grid
from .vander import _nbits
from .vander import _vander_flat_complex_kernel
from .vander import _vander_flat_kernel
from .vander import vander as _vander

logger = logging.getLogger(__name__)

# XPU3 per-program launch overhead dominates tiny kernels, so keep grid=1 with a
# tile sized to the actual numel: a 2048-wide tile for 16 useful elements wastes
# masked lanes, and splitting into many small programs costs more launch overhead
# than it hides.  Block stays a power of two (tl.arange requirement) with a 256
# floor (the generic op's effective sweet spot; 32/64-wide tiles were slower) and
# is capped by the kernel's native block, so large inputs run parallel programs.
_MIN_BLOCK = 256


def _pick_block(numel, max_block):
    return min(max_block, max(_MIN_BLOCK, triton.next_power_of_2(numel)))


def linalg_vander(x, N=None):
    """Kunlunxin override for ``aten::linalg_vander``.

    ``linalg_vander`` is the batched, float/complex-only cousin of ``vander``:
    ``out[..., i, j] = x[..., i] ** j`` for ``j in [0, N)`` (``N = x.shape[-1]``
    when omitted).  Flatten the leading dimensions into a single 1-D input of
    length ``prod(batch) * n`` and reuse the already-optimised ``vander`` flat-1D
    kernels (constexpr ``N``, flat contiguous DMA store, binary exponentiation)
    with ``increasing=True``.

    Deviations from simply calling ``vander``:

    * **Complex** — the generic op's complex kernel cannot compile on XPU3: its
      ``tl_extra_shim`` ``pow/atan2/cos/sin`` lower to ``"Unsupported"`` externs
      and the ``ConvertTritonXPUToLLVM`` pass fails.  The ``vander`` complex
      kernel builds powers by repeated ``(re, im)`` squaring, which is
      transcendental-free.  It is evaluated in **fp64** and only then rounded to
      the storage dtype: ``torch.linalg.vander``'s reference is exactly the fp64
      product rounded down, so a pure complex64 evaluation drifts ~1-3e-7 (XPU3
      silently downcasts ``complex128`` storage to ``complex64`` yet the test
      keeps the strict complex128 tolerance).
    * **fp32** — the plain flat kernel (not vander's double-float dd kernel).
      dd is exact but its heavily unrolled body makes XPU3's MLIR lowering blow
      up super-linearly (N=1024 takes >10 min in ``make_ttxir``), and it is
      slower than needed on tiny tiles.  Plain square-and-multiply stays well
      inside the fp32 tolerance (12k-trial sweep: worst |diff| is 0.44x the
      allowed atol+rtol*|ref|, worst rel 6.6e-7 < 1.3e-6).  ``N`` is constexpr so
      the bit loop unrolls to log2(N) cheap multiply/select steps.
    """
    logger.debug("GEMS_LINALG_VANDER (kunlunxin)")

    assert x.dtype.is_floating_point or x.dtype.is_complex, (
        f"Unsupported dtype {x.dtype}"
    )

    n = x.shape[-1]
    if N is None:
        N = n
    batch_dims = x.shape[:-1]
    final_shape = batch_dims + (n, N)

    # x[..., i] for i < prod(batch)*n lives contiguously on the flat input, and
    # the output is row-major with column index == exponent, which is exactly
    # vander(x_flat, N, increasing=True).
    x_flat = x.reshape(-1)
    m2 = x_flat.numel()
    numel = m2 * N

    if m2 == 0 or N == 0:
        return torch.empty(final_shape, dtype=x.dtype, device=x.device)

    nbits = _nbits(N)

    if x.is_complex():
        out = torch.empty(final_shape, dtype=x.dtype, device=x.device)
        xc = x_flat.contiguous()
        xr = xc.real.contiguous()
        xi = xc.imag.contiguous()
        outr = torch.empty((m2, N), dtype=xr.dtype, device=x.device)
        outi = torch.empty((m2, N), dtype=xr.dtype, device=x.device)
        _vander_flat_complex_kernel[_grid(numel, _BLOCK_COMPLEX)](
            xr,
            xi,
            outr,
            outi,
            numel,
            N=N,
            x_stride=xr.stride(0),
            increasing=True,
            NBITS=nbits,
            BLOCK=_BLOCK_COMPLEX,
            COMPUTE_DTYPE=tl.float64,
        )
        out_flat = out.reshape(-1)
        out_flat.real.copy_(outr.reshape(-1))
        out_flat.imag.copy_(outi.reshape(-1))
        return out

    if x.dtype == torch.float32:
        out = torch.empty(final_shape, dtype=x.dtype, device=x.device)
        block = _pick_block(numel, _BLOCK_FLOAT)
        _vander_flat_kernel[_grid(numel, block)](
            x_flat,
            out,
            numel,
            N=N,
            x_stride=x_flat.stride(0),
            increasing=True,
            NBITS=nbits,
            BLOCK=block,
            COMPUTE_DTYPE=tl.float32,
        )
        return out

    # fp16 / bf16 / fp64: reuse the tested vander path (fixed blocks).
    out = _vander(x_flat, N=N, increasing=True)
    return out.reshape(final_shape)
