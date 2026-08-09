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

from ..utils.codegen_config_utils import CodeGenConfig
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# A direct comparison with -inf has the same floating-point semantics as
# isinf(x) & (x < 0), but avoids the libdevice isinf extern call.
_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")], config=_config)
@triton.jit
def isneginf_func(x):
    return x.to(tl.float32) == -float("inf")


def isneginf(A):
    logger.debug("GEMS_KUNLUNXIN ISNEGINF")
    return isneginf_func(A)


def isneginf_out(A, *, out=None):
    logger.debug("GEMS_KUNLUNXIN ISNEGINF_OUT")
    if out is None:
        return isneginf_func(A)
    isneginf_func(A, out0=out)
    return out
