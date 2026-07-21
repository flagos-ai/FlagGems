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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def clip_func(x, min_val, max_val):
    x_fp32 = x.to(tl.float32)
    min_fp32 = min_val.to(tl.float32)
    max_fp32 = max_val.to(tl.float32)
    result = tl.minimum(tl.maximum(x_fp32, min_fp32), max_fp32)
    return result.to(x.dtype)


def clip(A, min_val=None, max_val=None):
    logger.debug("GEMS_ENFLAME CLIP")
    if min_val is None and max_val is None:
        return A.clone()
    if min_val is None:
        from .clamp import clamp

        return clamp(A, min=None, max=max_val)
    if max_val is None:
        from .clamp import clamp

        return clamp(A, min=min_val, max=None)
    return clip_func(A, min_val, max_val)


def clip_(A, min_val=None, max_val=None):
    logger.debug("GEMS_ENFLAME CLIP_")
    if min_val is None and max_val is None:
        return A
    if min_val is None:
        from .clamp import clamp_

        return clamp_(A, min=None, max=max_val)
    if max_val is None:
        from .clamp import clamp_

        return clamp_(A, min=min_val, max=None)
    clip_func(A, min_val, max_val, out0=A)
    return A
