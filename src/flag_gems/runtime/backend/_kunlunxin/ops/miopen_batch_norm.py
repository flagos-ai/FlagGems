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

from .batch_norm import batch_norm as _batch_norm

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
    """MIOpen batch normalization forward on Kunlunxin XPU.

    Delegates to the Kunlunxin ``batch_norm`` implementation. The generic
    ``flag_gems.ops.miopen_batch_norm`` launches the generic
    ``batch_norm_forward_kernel``, whose 2D-tile pointer arithmetic fails to
    lower on XPU ("tt.addptr" / "triton_xpu.convert_layout" verification
    failures, then uni_sram exhaustion). The vendor kernel has a different
    reduction layout that compiles and passes the whole forward matrix.

    Args:
        input: Input tensor
        weight: Weight tensor (scale)
        bias: Bias tensor
        running_mean: Running mean for inference
        running_var: Running variance for inference
        training: Whether to use training mode
        exponential_average_factor: Momentum for running mean/var updates
        epsilon: Small value for numerical stability

    Returns:
        Tuple of (output, save_mean, save_invstd)
    """
    logger.debug("GEMS_KUNLUNXIN MIOPEN_BATCH_NORM_FORWARD")

    # exponential_average_factor is used as momentum; the vendor entry point
    # expects the running-stat buffers to be real tensors even when unused.
    running_mean = input if running_mean is None else running_mean
    running_var = input if running_var is None else running_var

    output, mean, inv_std = _batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training=training,
        momentum=exponential_average_factor,
        eps=epsilon,
        # aten::miopen_batch_norm updates the running statistics in every
        # dtype, and its running variance is the unbiased (1/(N-1)) estimate,
        # as on CPU and CUDA. See the module docstring.
        update_running_all_dtypes=True,
        unbiased_running_var=True,
    )

    return output, mean, inv_std
