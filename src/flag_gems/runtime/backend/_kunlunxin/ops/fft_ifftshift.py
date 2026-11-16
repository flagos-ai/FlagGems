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

"""Kunlunxin XPU3 override for ``aten::fft_ifftshift``.

``ifftshift`` is a pure data permutation: along every selected dimension it is a
roll by ``size // 2``.  The generic KernelGen kernel walks the tensor element by
element, recomputing a full multi-dimensional index (int64 ``div``/``mod`` from
scalar ``sizes``/``strides`` buffers) and copying one *byte* at a time.  On XPU3
that is both a soft-integer-division wall and a byte-wide gather/scatter, and the
host side additionally materialises four int64 metadata tensors plus one
``.item()`` device sync per dimension on every call (~200 ms fixed floor).

Roll along one dimension decomposes into exactly two strided sub-copies::

    out.narrow(d, 0, h) = x.narrow(d, s, h)     # h = size - size // 2
    out.narrow(d, h, s) = x.narrow(d, 0, s)     # s = size // 2

Both views keep the innermost stride at 1 on each side, which is exactly the
shape the Kunlunxin ``tle.dsa`` row-DMA descriptor handles: one contiguous run
per row, independent (identical) src/dst row strides.  The backend already
carries that primitive as ``utils/tle_copy.tle_copy`` (a real XPU DMA kernel, no
aten dispatch).  Multiple selected dimensions commute, so they are applied as a
sequence of single-dimension rolls through ping-pong buffers.

If ``tle_copy`` cannot express a layout (tiny/odd runs, unaligned contiguous
base, unsupported dtype, ...) the operator falls back to the generic Triton
kernel for correctness -- never to aten/CPU.
"""

import logging

import torch

from flag_gems.ops.fft_ifftshift import fft_ifftshift as _generic_ifftshift

try:
    from ..utils.tle_copy import tle_copy, tle_dma_available
except ImportError:  # triton build without the XPU tile-language extension
    tle_copy = None
    tle_dma_available = lambda: False  # noqa: E731

logger = logging.getLogger(__name__)


def _normalize_dims(ndim, dim):
    if dim is None:
        return list(range(ndim))
    dims = [dim] if isinstance(dim, int) else list(dim)
    dims = [(d + ndim) % ndim for d in dims]
    seen = set()
    out = []
    for d in dims:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def fft_ifftshift(self, dim=None):
    logger.debug("GEMS_KUNLUNXIN FFT_IFFTSHIFT")

    x = self
    out = torch.empty_like(x)

    # Empty outputs carry no data; the generic clone semantics are reproduced by
    # the (uninitialised) empty_like shape alone.
    if x.numel() == 0:
        return out

    # A non-contiguous input cannot be sliced into unit-stride DMA rows; keep the
    # already-correct generic Triton kernel.
    if not x.is_contiguous() or not tle_dma_available():
        return _generic_ifftshift(x, dim)

    dims = _normalize_dims(x.ndim, dim)
    # Dimensions of extent <= 1 are a no-op roll (shift == 0); if none of the
    # selected dimensions actually moves data the whole tensor is just copied.
    effective = [d for d in dims if x.shape[d] > 1]

    if not effective:
        if tle_copy(x, out):
            return out
        return _generic_ifftshift(x, dim)

    cur = x
    last = len(effective) - 1
    for i, d in enumerate(effective):
        dst = out if i == last else torch.empty_like(x)
        n = x.shape[d]
        s = n // 2
        h = n - s
        ok = tle_copy(cur.narrow(d, s, h), dst.narrow(d, 0, h)) and tle_copy(
            cur.narrow(d, 0, s), dst.narrow(d, h, s)
        )
        if not ok:
            return _generic_ifftshift(x, dim)
        cur = dst

    return out
