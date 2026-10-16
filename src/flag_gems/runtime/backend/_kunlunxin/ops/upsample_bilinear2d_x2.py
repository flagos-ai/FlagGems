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

"""tle.raw 2x bilinear upsampling, shared by ``upsample_bilinear2d`` and
``upsample_bilinear2d_aa``.

At an exact 2x scale the antialiased triangular filter degenerates to the
plain two-tap bilinear filter, so both operators can use the same payload
``upsample_bilinear2d_x2.xpu`` (verified bit-exact on CPU).

The payload is a hand-written XTDK cluster kernel: one program per cluster,
64 cores per cluster, contiguous GM2LM/LM2GM only, a 3-row sliding window
held in fp32 local memory.  See the header of the ``.xpu`` file for the
schedule.

``upsample_bilinear2d_x2`` returns ``None`` when the fast path does not apply
(any other scale, ``align_corners=True``, explicit scales, non-contiguous or
unsupported dtype), so callers fall back to their generic kernels.
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Number of lanes per program.  Must stay 64 -- the payload's column tiling
# assumes `core_num()` cores cooperate inside one cluster.
LANES = 64
# Tile width (input columns) baked into the payload instantiations.
TILE_W = 128
# Clusters to launch (one program == one cluster).  Each cluster adds a full
# set of `LANES` cores of DMA + ALU throughput; the payload folds
# `cluster_id()` into its global core index.  The P800 has 12 clusters
# (= 768 cores); launching more only adds waves.
NCLUSTER = int(os.environ.get("FG_UB2D_CLUSTERS", "12"))

_HAS_TLE_RAW = False
try:
    import triton.experimental.tle as _tle_ext
    import triton.experimental.tle.language as _tle_lang

    if hasattr(_tle_ext, "raw") and hasattr(_tle_ext.raw, "dialect"):
        _XPU_FILE = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "upsample_bilinear2d_x2.xpu"
        )

        @_tle_ext.raw.dialect("xpu3", file=_XPU_FILE)
        def upsample_bilinear2d_x2_f32(inp, out, NC, IH, IW):
            ...

        @_tle_ext.raw.dialect("xpu3", file=_XPU_FILE)
        def upsample_bilinear2d_x2_f16(inp, out, NC, IH, IW):
            ...

        @_tle_ext.raw.dialect("xpu3", file=_XPU_FILE)
        def upsample_bilinear2d_x2_bf16(inp, out, NC, IH, IW):
            ...

        # The scalar params must stay runtime `i32` operands: tle.raw.call only
        # accepts tensors, and Triton would otherwise specialise values equal
        # to 1 into constexprs.
        @triton.jit(do_not_specialize=["NC", "IH", "IW"])
        def _ub2d_x2_kernel_f32(INP, OUT, NC, IH, IW):
            _tle_lang.raw.call(upsample_bilinear2d_x2_f32, (INP, OUT, NC, IH, IW))

        @triton.jit(do_not_specialize=["NC", "IH", "IW"])
        def _ub2d_x2_kernel_f16(INP, OUT, NC, IH, IW):
            _tle_lang.raw.call(upsample_bilinear2d_x2_f16, (INP, OUT, NC, IH, IW))

        @triton.jit(do_not_specialize=["NC", "IH", "IW"])
        def _ub2d_x2_kernel_bf16(INP, OUT, NC, IH, IW):
            _tle_lang.raw.call(upsample_bilinear2d_x2_bf16, (INP, OUT, NC, IH, IW))

        _KERNELS = {
            torch.float32: _ub2d_x2_kernel_f32,
            torch.float16: _ub2d_x2_kernel_f16,
            torch.bfloat16: _ub2d_x2_kernel_bf16,
        }
        _HAS_TLE_RAW = True
except Exception:  # noqa: BLE001
    _KERNELS = {}


def upsample_bilinear2d_x2(
    input, output_size, align_corners, scales_h=None, scales_w=None
):
    """2x bilinear upsample via the tle.raw payload, or ``None`` if unsupported."""
    if not _HAS_TLE_RAW:
        return None
    kernel = _KERNELS.get(input.dtype)
    if kernel is None:
        return None
    if input.ndim != 4 or not input.is_contiguous():
        return None
    if align_corners or scales_h is not None or scales_w is not None:
        return None

    n, c, ih, iw = input.shape
    oh, ow = int(output_size[0]), int(output_size[1])
    if oh != 2 * ih or ow != 2 * iw:
        return None
    # The payload drops columns when the input row does not fit in
    # NCT * TILE_W <= LANES * TILE_W.
    if iw > LANES * TILE_W:
        return None
    if n == 0 or c == 0 or ih == 0 or iw == 0:
        return None

    out = torch.empty((n, c, oh, ow), device=input.device, dtype=input.dtype)
    with torch_device_fn.device(input.device):
        kernel[(NCLUSTER,)](input, out, n * c, ih, iw)
    return out
