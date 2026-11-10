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

from flag_gems.ops.div import true_divide as _pointwise_true_divide
from flag_gems.ops.div import true_divide_ as _pointwise_true_divide_

from .copy import copy_

logger = logging.getLogger(__name__)


def _divide_inplace(A, B):
    dtype = torch.result_type(A, B)
    if not (dtype.is_floating_point or dtype.is_complex):
        dtype = torch.get_default_dtype()
    if not torch.can_cast(dtype, A.dtype):
        raise RuntimeError(
            f"result type {dtype} cannot be cast to inplace dtype {A.dtype}"
        )
    if isinstance(B, torch.Tensor):
        if torch.broadcast_shapes(A.shape, B.shape) != A.shape:
            raise RuntimeError("division result shape does not match inplace shape")
        direct = B.shape == A.shape and B.is_contiguous() and not B.is_complex()
    else:
        direct = not isinstance(B, complex)
    if torch._debug_has_internal_overlap(A) == 1:
        raise RuntimeError("inplace division does not support overlapping output")
    if A.numel() == 0:
        return A
    if direct and A.is_contiguous() and not A.is_complex():
        return _pointwise_true_divide_(A, B)
    # The generated strided pointwise kernel cannot safely alias its input on
    # Ascend. Its complex dispatcher also returns a fresh tensor, ignoring out0.
    # Compute first, then copy to preserve the original view and return object.
    result = _pointwise_true_divide(A, B)
    if A.is_complex():
        # Component views keep complex writeback on the Triton copy path.
        copy_(torch.view_as_real(A), torch.view_as_real(result))
    else:
        copy_(A, result)
    return A


def true_divide_(A, B):
    logger.debug("GEMS_ASCEND TRUE_DIVIDE_")
    return _divide_inplace(A, B)


def true_divide_tensor_(A, B):
    logger.debug("GEMS_ASCEND TRUE_DIVIDE_")
    return _divide_inplace(A, B)
