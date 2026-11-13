# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.runtime import torch_device_fn

from .softmax import softmax as _softmax_kunlunxin

try:
    import triton
    import triton.experimental.tle.language as tle
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_TLE = True
except ImportError:  # triton without the XPU tile-language extension
    _HAS_TLE = False

logger = logging.getLogger(__name__)

# The cluster DMA is fastest when one program stages ~128 KB; beyond ~64 rows
# per program the per-row reduce result write and the launch tail dominate.
_SP_TARGET_BYTES = 128 * 1024
_SP_MAX_XBLOCK = 64
# 4 KB of local memory per core over 64 cores bounds one program's tile.
_SP_LM_BUDGET = 256 * 1024
# Below this one program's tile is too small for the DMA to pay for itself.
_SP_MIN_BYTES = 32 * 1024
# Below one full wave of cores the tile staging never amortises the launch.
_SP_MIN_PROGRAMS = 64
_SP_GEOM = {}

if _HAS_TLE:
    _TL_DTYPE = {
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
    }

    @triton.jit(
        do_not_specialize=["N"],
        do_not_specialize_on_alignment=["a_desc", "out_desc"],
    )
    def _softmax_lm_kernel(
        a_desc,
        out_desc,
        N,
        XBLOCK: tl.constexpr,
        YBLOCK: tl.constexpr,
        IN_DTYPE: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        """Last-axis softmax with the whole row resident in local memory.

        The row is staged GM -> LM once, both reductions run on the LM-resident
        tile, and the normalised tile goes back out through the same buffer, so
        the row is read from GM exactly once. ``out = e * (1 / z)`` rather than
        ``e / z``: the reciprocal is hoisted to one per row.
        """
        pid = tl.program_id(0)
        row_off = pid * XBLOCK
        a = tle.gpu.alloc(
            [XBLOCK, YBLOCK], dtype=IN_DTYPE, layout=None, scope=tle.gpu.lmem
        )
        rows = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
        cols = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
        ptrs = tle.gpu.local_ptr(a, (rows, cols))
        tle.gpu.copy(a_desc, a, [XBLOCK, YBLOCK], [row_off, 0])
        x = tl.load(ptrs).to(tl.float32)
        m = tl.max(x, 1)
        e = tl.exp(x - m[:, None])
        z = tl.sum(e, 1)
        tl.store(ptrs, (e * (1.0 / z)[:, None]).to(OUT_DTYPE))
        tle.gpu.copy(a, out_desc, [XBLOCK, YBLOCK], [row_off, 0])


def _sp_geom(M, N, itemsize):
    """Rows per program for a contiguous [M, N] last-axis softmax, or None."""
    key = (M, N, itemsize)
    if key in _SP_GEOM:
        return _SP_GEOM[key]
    xblock = None
    if N >= 1:
        xblock = min(_SP_MAX_XBLOCK, _SP_TARGET_BYTES // (N * itemsize))
        xblock = 1 << (xblock.bit_length() - 1) if xblock > 0 else 1
        while xblock > 1 and M % xblock:
            xblock >>= 1
        if (
            xblock * N * itemsize > _SP_LM_BUDGET
            or xblock * N * itemsize < _SP_MIN_BYTES
            or M // xblock < _SP_MIN_PROGRAMS
        ):
            # Too few rows left after the divisibility reduction: the tile no
            # longer carries a worthwhile DMA and the pointer path is faster.
            xblock = None
    _SP_GEOM[key] = xblock
    return xblock


def _sp_try(inp, dim):
    """Run the LM-resident kernel; None when the shape is not expressible."""
    if not _HAS_TLE:
        return None
    if inp.dtype not in _TL_DTYPE:
        return None
    ndim = inp.ndim
    if dim < 0:
        dim += ndim
    if dim != ndim - 1 or not inp.is_contiguous():
        return None
    N = inp.shape[-1]
    if N < 2 or N & (N - 1):
        # YBLOCK must be a power of two for the tile address arithmetic.
        return None
    M = inp.numel() // N
    xblock = _sp_geom(M, N, inp.element_size())
    if xblock is None:
        return None
    out = torch.empty_like(inp)
    a = inp.view(M, N)
    c = out.view(M, N)
    device = inp.device
    with torch_device_fn.device(device):
        _softmax_lm_kernel[(M // xblock,)](
            TensorDescriptor.from_tensor(a, block_shape=[xblock, N]),
            TensorDescriptor.from_tensor(c, block_shape=[xblock, N]),
            N,
            XBLOCK=xblock,
            YBLOCK=N,
            IN_DTYPE=_TL_DTYPE[inp.dtype],
            OUT_DTYPE=_TL_DTYPE[out.dtype],
        )
    return out


def special_softmax(self, dim, dtype=None):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if dtype is not None:
        self = self.to(dtype)

    out = _sp_try(self, dim)
    if out is not None:
        return out
    return _softmax_kunlunxin(self, dim)
