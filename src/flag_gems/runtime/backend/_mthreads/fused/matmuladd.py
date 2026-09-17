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

from flag_gems.runtime.backend._mthreads.ops.addmm import _addmm_impl

logger = logging.getLogger(__name__)


def matmuladd(input, other, bias):
    """
    Matrix multiplication with addition: output = matmul(input, other) + bias

    Routes to the mthreads addmm implementation: fp16/bf16 compatible
    layouts go through the SQMMA TensorDescriptor kernel, and fp32 (or
    non-SQMMA shapes) through the FMA kernel, with alpha=1, beta=1.
    """
    logger.debug("GEMS_MTHREADS MATMULADD")
    return _addmm_impl(bias, input, other, out=None, beta=1, alpha=1)
