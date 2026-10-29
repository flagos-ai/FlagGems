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
from dataclasses import replace

import triton
import triton.language as tl

from flag_gems.runtime import device

from ..utils.codegen_config_utils import get_codegen_config
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name

# minimum is purely bandwidth bound, and the async-copy pipeline the default
# config enables costs more in issue overhead than it recovers in overlap here:
# on P800 it leaves fp32 at 1517 GB/s and fp16 at 923 GB/s, while the
# synchronous path reaches 1752 GB/s (sp 0.74 -> 0.86) and 1265 GB/s
# (sp 0.47 -> 0.64). bf16 improves too (0.28 -> 0.33) but stays low: the XPU
# backend widens bf16 to 32-bit on load, so the compare costs ~2.8x its fp16
# equivalent even in a hand-written kernel.
_minimum_config = replace(get_codegen_config(), isCloseMemoryAsync=False)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 0, "DEFAULT")],
    config=_minimum_config,
)
@triton.jit
def minimum_kernel(X, Y):
    if X.dtype == tl.bfloat16:
        X = X.to(tl.float32)
        Y = Y.to(tl.float32)
    return tl.minimum(X, Y)


def minimum(X, Y):
    logger.debug("GEMS_KUNLUNXIN MINIMUM")
    assert X.device.type == device and Y.device.type == device
    return minimum_kernel(X, Y)
