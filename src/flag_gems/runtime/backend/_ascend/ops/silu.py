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

from flag_gems.runtime.backend._ascend.utils import CORE_NUM
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

logger = logging.getLogger(__name__)


silu_config = CodeGenConfig(
    max_tile_size=4096,
    max_grid_size=(CORE_NUM, 1, 1),
    max_num_warps_per_cta=32,
    prefer_block_pointer=False,
    prefer_1d_tile=int(triton.__version__[0]) < 3,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=silu_config)
@triton.jit
def silu_forward(x):
    x_fp32 = x.to(tl.float32)
    return tl.fdiv(x_fp32, 1.0 + tl.exp(-x_fp32))


def silu(self):
    logger.debug("GEMS_ASCEND SILU FORWARD")
    if self.is_contiguous():
        return silu_forward(self)
    # Ascend empty_like may not preserve a transposed input's strides. Providing
    # the output explicitly keeps pointwise_dynamic on its strided indexing path.
    output = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    return silu_forward(self, out0=output)


def silu_(self):
    logger.debug("GEMS_ASCEND SILU_ FORWARD")
    return silu_forward(self, out0=self)
