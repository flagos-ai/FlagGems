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

from flag_gems.runtime import device

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

# maximum is purely bandwidth bound, so the whole win is in the launcher's tile
# width and whether loads are issued asynchronously.  Two findings on P800:
#
# 1. Async loads (isCloseMemoryAsync=False) are worth ~0.11 PerfAvg on their own.
#
# 2. Tile 24576 is the fastest tile for both 2- and 4-byte elements.  The
#    launcher derives the tile as cdiv(buffer_size_limit * 64, element_size)
#    capped at 32768, and then rewrites an exactly-32768 tile down to 16384 to
#    avoid a device hang (_BAD_TILE_SIZE_1D in pointwise_dynamic).  So asking
#    for the full 32768 is counterproductive -- it lands back on 16384 -- while
#    a buffer that yields 24576 keeps the wider tile.  Reaching 24576 needs a
#    different buffer per element size, hence the two configs below:
#
#      element_size  buffer  tile    4096x4096 speedup (was, at tile 16384)
#      2 bytes       768     24576   fp16 0.85 (0.50), bf16 0.36 (0.30)
#      4 bytes       1536    24576   fp32 0.98 (0.77)
#
# buffer 640/896 (tiles 20480/28672) fail to compile in ConvertTritonXPUToLLVM,
# so 768 is the largest usable 2-byte setting.  Leave vectorization alone:
# isCloseVectorization=True measured ~3.5x slower here.
#
# 3. kunlunAutoGrid keeps a small tensor on one cluster, which is worth ~0.15 on
#    the 64x64 shapes -- but its single-cluster grid reads out of bounds on a
#    non-contiguous input: `torch.maximum(x.t(), y)` on a 128x128 fp32 tensor
#    dies with an illegal memory access.  It is only enabled for plain
#    contiguous same-shape inputs, which is what the benchmark and the common
#    call path use; everything else keeps the shipped 12-cluster grid.  The
#    contiguous path was stress-checked against torch over 32 shapes x 3 dtypes
#    (1 element up to 1M, including sizes straddling the cluster threshold).
_BASE_CFG = dict(
    max_tile_size=512,
    max_grid_size=(65536, 65536, 65536),
    max_num_warps_per_cta=32,
    prefer_block_pointer=True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
)

config_narrow_ = CodeGenConfig(buffer_size_limit=768, **_BASE_CFG)
config_wide_ = CodeGenConfig(buffer_size_limit=1536, **_BASE_CFG)
config_narrow_grid_ = CodeGenConfig(
    buffer_size_limit=768, kunlunAutoGrid=True, **_BASE_CFG
)
config_wide_grid_ = CodeGenConfig(
    buffer_size_limit=1536, kunlunAutoGrid=True, **_BASE_CFG
)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_narrow_,
)
@triton.jit
def maximum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)

    return tl.maximum(X, Y)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_wide_,
)
@triton.jit
def maximum_kernel_wide(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)

    return tl.maximum(X, Y)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_narrow_grid_,
)
@triton.jit
def maximum_kernel_grid(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)

    return tl.maximum(X, Y)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_wide_grid_,
)
@triton.jit
def maximum_kernel_wide_grid(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)

    return tl.maximum(X, Y)


def maximum(X, Y):
    logger.debug("GEMS_KUNLUNXIN MAXIMUM")
    assert X.device.type == device and Y.device.type == device
    # Pick whichever config derives a 24576-element tile for this element size.
    # The output dtype drives the tile, so promote first for mixed inputs.
    wide = torch.promote_types(X.dtype, Y.dtype).itemsize > 2
    # The autoGrid variants are only safe for plain contiguous same-shape
    # inputs (see note 3 above).
    if X.shape == Y.shape and X.is_contiguous() and Y.is_contiguous():
        return maximum_kernel_wide_grid(X, Y) if wide else maximum_kernel_grid(X, Y)
    return maximum_kernel_wide(X, Y) if wide else maximum_kernel(X, Y)
