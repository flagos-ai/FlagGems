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

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

_isnan = tl_extra_shim.isnan
logger = logging.getLogger(__name__)

# nan_to_num is an elementwise select (isnan / ±inf checks + tl.where): a pure
# memory-bound copy/select. The old kunlunxin override used a BARE pointwise_dynamic
# with NO CodeGenConfig, so on XPU it fell to the default path (buffer_size_limit
# 2048, no kunlunAutoGrid, no unroll) -> BLOCK=512 1d tile, underutilized bandwidth
# (see harness/perf_ir_3/ir-nan_to_num-dev0.log). Reuse the proven memory-bound
# select/copy recipe shared by neg / view_copy / masked_fill (autoGrid + unroll8 +
# vec-open + buffer_size_limit=4096). Kernel body unchanged (zero correctness risk).
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
    x_nan = _isnan(x.to(tl.float32))
    x_posinf = x == float("inf")
    x_neginf = x == -float("inf")
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
