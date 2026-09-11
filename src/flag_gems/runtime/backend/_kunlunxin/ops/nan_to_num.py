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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# nan_to_num is an elementwise select (isnan / ±inf checks + tl.where): a pure
# memory-bound select/copy. Two independent findings drive this implementation:
#
# 1. Config: the old kunlunxin override used a BARE pointwise_dynamic with NO
#    CodeGenConfig, so on XPU it fell to the default path (buffer_size_limit
#    2048, no kunlunAutoGrid, no unroll) -> BLOCK=512 1d tile, underutilized
#    bandwidth. This reuses the proven memory-bound select/copy recipe shared
#    by neg / view_copy / masked_fill (autoGrid + unroll8 + buffer 4096).
#    Config sweep confirmed unroll16/buffer8192 and isCloseVectorization=True
#    give no further gain on this kernel.
#
# 2. NaN detection: `_isnan(x.to(tl.float32))` (extern libdevice call) is the
#    dominant cost on XPU — extern_elementwise lowers to a scalar/throughput-
#    limited path (~10x slower than a pure select, ~63 GB/s at [4096,4096]
#    fp16). Replaced it with an integer bit trick on the fp32 bits:
#      NaN    := (bits & 0x7FFFFFFF) > 0x7F800000   (exponent all-ones, mantissa != 0)
#      +inf   := bits == 0x7F800000
#      -inf   := bits == 0xFF800000
#    This is exact IEEE-754 semantics (bit identities for NaN/inf are unique),
#    uses only cheap integer ALU ops and removes the extern call. fp32 dtype
#    path needs no conversion at all; fp16/bf16 pay the same single fp32
#    upcast as before but skip the extern. Bit-identical output.
#    Self-compare variants (x != x / x > x) crash the XPU llir pass
#    (PassManager::run failed) and int16 bitcast is unsupported — both dead
#    ends; the fp32 bitmask is the fastest verified body.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False, False, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    # IEEE-754 bit patterns (fp32): |inf| = 0x7F800000, any NaN has exponent
    # all-ones and nonzero mantissa, so (bits & 0x7FFFFFFF) > 0x7F800000 is
    # exactly isnan; and ==0x7F800000 / ==0xFF800000 are +/-inf.
    x_bits = x.to(tl.float32).to(tl.int32, bitcast=True)
    x_nan = (x_bits & 0x7FFFFFFF) > 0x7F800000
    x_posinf = x_bits == 0x7F800000
    x_neginf = x_bits == 0xFF800000
    x = tl.where(x_nan, nan, x)
    x = tl.where(x_posinf, posinf, x)
    x = tl.where(x_neginf, neginf, x)
    return x


# nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_KUNLUNXIN NAN_TO_NUM")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    return nan_to_num_func(A, nan, posinf, neginf)
