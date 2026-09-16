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

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isneginf_func(x):
    x_fp32 = x.to(tl.float32)
    return tl_extra_shim.isinf(x_fp32) & (x_fp32 < 0)


def isneginf(A):
    logger.debug("GEMS ISNEGINF")
    return isneginf_func(A)


def isneginf_out(A, *, out=None):
    logger.debug("GEMS ISNEGINF_OUT")
    if out is None:
        return isneginf_func(A)
    isneginf_func(A, out0=out)
    return out
