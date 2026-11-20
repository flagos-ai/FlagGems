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

import flag_gems
from flag_gems.runtime import device
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)
device = device.name


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func(x, y):
    # Compare in a dtype that does not discard information: FP32 represents
    # fp16/bf16 exactly, but it rounds integers above 2**24 (so distinct ids
    # can compare equal) and truncates fp64 values.  Integer operands must
    # therefore stay in an integer type, and fp64 must stay in fp64.
    if x.dtype.is_fp64() or y.dtype.is_fp64():
        return x.to(tl.float64) == y.to(tl.float64)
    elif x.dtype.is_floating() or y.dtype.is_floating():
        return x.to(tl.float32) == y.to(tl.float32)
    else:
        return x.to(tl.int64) == y.to(tl.int64)


def eq(A, B):
    if A.device != B.device:
        if A.device.type == device:
            B = B.to(A.device)
        else:
            A = A.to(B.device)
    logger.debug("GEMS EQ")
    return eq_func(A, B)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def eq_func_scalar(x, y):
    # A floating-point scalar promotes the comparison to floating point (it
    # must not be truncated to an integer), while an integer scalar keeps
    # integer semantics so that large integer values stay exact.
    if y.dtype.is_floating() or x.dtype.is_floating():
        if x.dtype.is_fp64() or y.dtype.is_fp64():
            return x.to(tl.float64) == y.to(tl.float64)
        return x.to(tl.float32) == y.to(tl.float32)
    return x.to(tl.int64) == y.to(tl.int64)


def eq_scalar(A, B):
    logger.debug("GEMS EQ_SCALAR")
    return eq_func_scalar(A, B)


def equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    logger.debug("GEMS EQUAL")
    if x.shape != y.shape:
        return False
    eq_tensor = eq(x, y)
    return bool(flag_gems.all(eq_tensor).item())
