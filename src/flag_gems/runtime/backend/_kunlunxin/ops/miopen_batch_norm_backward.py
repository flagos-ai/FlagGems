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

from torch import Tensor

from .batch_norm import batch_norm_backward

logger = logging.getLogger(__name__)


def miopen_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean=None,
    running_var=None,
    save_mean=None,
    save_var=None,
    epsilon: float = 1e-05,
) -> tuple:
    """Backward pass for batch normalization (MIOpen variant) on Kunlunxin XPU.

    The MIOpen schema calls the saved inverse standard deviation argument
    ``save_var``. This override delegates to the Kunlunxin ``batch_norm_backward``
    kernel path: the generic implementation in ``flag_gems.ops.miopen_batch_norm_backward``
    relies on the generic ``batch_norm_backward_kernel`` whose 2D-tile lowering fails on
    XPU ("triton_xpu.convert_layout" shape mismatch), while the vendor kernel
    (transposed [N*S, C, 1] view + per-feature grid) compiles and passes the whole
    batch-norm backward matrix.

    Returns:
        Tuple of (grad_input, grad_weight, grad_bias).
    """
    logger.debug("GEMS_KUNLUNXIN MIOPEN_BATCH_NORM_BACKWARD")
    return batch_norm_backward(
        grad_output,
        input,
        weight=weight,
        running_mean=running_mean,
        running_var=running_var,
        save_mean=save_mean,
        save_invstd=save_var,
        train=True,
        eps=epsilon,
        output_mask=(True, True, True),
    )