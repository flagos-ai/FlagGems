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
import triton.language as tl

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isfinite_func(x):
    if x.dtype.is_fp64():
        return (x.to(tl.int64, bitcast=True) & 0x7FFFFFFFFFFFFFFF) < 0x7FF0000000000000
    elif x.dtype.is_fp32():
        return (x.to(tl.int32, bitcast=True) & 0x7FFFFFFF) < 0x7F800000
    elif x.dtype.is_fp16():
        return (x.to(tl.int16, bitcast=True) & 0x7FFF) < 0x7C00
    elif x.dtype.is_bf16():
        return (x.to(tl.int16, bitcast=True) & 0x7FFF) < 0x7F80


def isfinite(
    A: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_CAMBRICON ISFINITE")
    if A.is_floating_point():
        legal_dtype = [torch.float32, torch.float16, torch.bfloat16]
        assert (
            A.dtype in legal_dtype
        ), f"isfinite input float dtype should in {str(legal_dtype)}, get {str(A.dtype)}"
        return isfinite_func(A)
    else:
        return torch.full(A.shape, True, dtype=torch.bool, device=A.device)
