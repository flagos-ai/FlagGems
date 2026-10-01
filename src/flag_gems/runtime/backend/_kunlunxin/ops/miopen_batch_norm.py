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
from typing import Optional, Tuple

from torch import Tensor

from .native_batch_norm import native_batch_norm

logger = logging.getLogger(__name__)


def miopen_batch_norm(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Forward pass for batch normalization (MIOpen variant) on Kunlunxin XPU.

    The generic implementation in ``flag_gems.ops.miopen_batch_norm`` binds the
    shared ``batch_norm_forward_kernel`` (2D-tile Welford), whose lowering fails
    on XPU ("triton_xpu.convert_layout" op requires the same shape ... during
    ``TritonXPUUnrollControl``). This override delegates to the Kunlunxin
    ``native_batch_norm`` kernel path, which is the exact operator used as the
    test reference and compiles/runs on XPU. The MIOpen schema is a 1:1 rename of
    ``native_batch_norm`` (``exponential_average_factor`` == momentum,
    ``epsilon`` == eps); the returned ``save_var`` is the saved inverse standard
    deviation, matching the generic convention and the backward override.

    Returns:
        Tuple of (output, save_mean, save_var(=inv_std)).
    """
    logger.debug("GEMS_KUNLUNXIN MIOPEN_BATCH_NORM FORWARD")
    return native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )
