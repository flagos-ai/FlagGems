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

import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

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


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def hardswish_func(x):
    # hardswish(x) = x * relu6(x + 3) / 6 = x * min(max(x + 3, 0), 6) / 6
    # Compute in fp32 so fp16/bf16 inputs match PyTorch's accumulation precision.
    xf = x.to(tl.float32)
    inner = tl.minimum(tl.maximum(xf + 3.0, 0.0), 6.0)
    y = xf * inner * (1.0 / 6.0)
    return y.to(x.dtype)


def hardswish_(self):
    logger.debug("GEMS_KUNLUNXIN HARDSWISH_")
    out = hardswish_func(self, out0=self)
    return out
