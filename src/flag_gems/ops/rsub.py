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

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def rsub_func(x, y, alpha):
    return y - x * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def rsub_func_tensor_scalar(x, y, alpha):
    return y - x * alpha


_ALPHA_MUST_NOT_BE_FLOAT = (
    "For integral input tensors, argument alpha must not be a floating point number."
)


def _check_alpha(alpha, integral: bool):
    """ATen rejects a floating alpha for integral inputs; mirror that instead of
    silently truncating alpha inside the kernel."""
    if not integral:
        return
    if isinstance(alpha, float) and float(alpha) != int(alpha):
        raise RuntimeError(_ALPHA_MUST_NOT_BE_FLOAT)


def rsub_tensor(A, B, *, alpha=1):
    logger.debug("GEMS RSUB_TENSOR")
    _check_alpha(
        alpha, A.dtype.is_floating_point is False and A.dtype.is_complex is False
    )
    return rsub_func(A, B, alpha)


def rsub_scalar(A, B, alpha=1):
    logger.debug("GEMS RSUB_SCALAR")
    _check_alpha(
        alpha,
        isinstance(A, torch.Tensor)
        and A.dtype.is_floating_point is False
        and A.dtype.is_complex is False,
    )
    return rsub_func_tensor_scalar(A, B, alpha)
