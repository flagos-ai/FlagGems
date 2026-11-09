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

from flag_gems.ops.copy import copy_ as _generic_copy_

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=(True,), promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy(src):
    return src


def copy_(*args, **kwargs):
    # aten's `copy_` takes `(dst, src)`, so the single-argument pointwise kernel
    # that used to sit here raised "The lists of inputs and is_tensor must have
    # the same length" on every call. The generic implementation already handles
    # broadcasting, dtype conversion and aliasing, so defer to it.
    logger.debug("GEMS_ENFLAME COPY_")
    return _generic_copy_(*args, **kwargs)
