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
from typing import List, Optional, Tuple, Union

import torch

from .cat import cat, cat_out

logger = logging.getLogger(__name__)


def _reshape_input(t: torch.Tensor) -> torch.Tensor:
    # Zero or one dimensional tensors are reshaped into a (numel, 1) column.
    if t.ndim <= 1:
        return t.reshape(t.numel(), 1)
    return t


def column_stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COLUMN_STACK")

    if len(tensors) == 0:
        raise RuntimeError("column_stack expected a non-empty TensorList")

    reshaped = [_reshape_input(t) for t in tensors]

    # Type promotion: find the common dtype for all tensors (cat assumes a
    # single dtype and does not promote).
    dtype = reshaped[0].dtype
    for t in reshaped[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    reshaped = [t.to(dtype) if t.dtype != dtype else t for t in reshaped]

    # column_stack is exactly a concatenation along dim=1 after the reshape, so
    # delegate to the kunlunxin cat override, whose dim>0 path copies each
    # contiguous input through the vendor strided-copy engine
    # (torch.ops.aten._copy_from into a narrow view), which saturates the copy
    # bandwidth instead of hitting the block-DMA ~555GB/s ceiling of a triton
    # strided store.
    if out is None:
        return cat(reshaped, dim=1)
    return cat_out(reshaped, dim=1, out=out)


def column_stack_out(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN COLUMN_STACK_OUT")
    return column_stack(tensors, out=out)
