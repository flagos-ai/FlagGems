import logging

import triton
import triton.language as tl  # noqa: F401

from flag_gems.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

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


logger = logging.getLogger(__name__)


@pointwise_dynamic(
    config=CodeGenConfig(16384, (48, 1, 1), 4, True, False, False),
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
)
@triton.jit
def le_func(x, y):
    # Compare directly in the original dtype: <= is exact, no upcast to float32 needed;
    # after dropping one f16/bf16 -> f32 conversion, bandwidth utilization improves noticeably for small
    # tensors (e.g. [1024,16]) and medium tensors (GRID widened to 12 SIP * 4 CTA = 48 programs).
    return x <= y


def le(A, B):
    logger.debug("GEMS_ENFLAME LE")
    return le_func(A, B)


@pointwise_dynamic(
    config=CodeGenConfig(16384, (48, 1, 1), 4, True, False, False),
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
)
@triton.jit
def le_func_scalar(x, y):
    return x <= y


def le_scalar(A, B):
    logger.debug("GEMS_ENFLAME LE_SCALAR")
    return le_func_scalar(A, B)
