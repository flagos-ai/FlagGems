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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    # unroll_num=8 breaks the bool-op buffer into unrolled chunks so the
    # monolithic 2048-wide llvm.struct is never formed (avoids ~16GB IR);
    # isCloseMemoryAsync stays at its default True (see bitwise_and).
    kunlunAutoGrid=True,
    unroll_num=8,
)


# int1 path: direct logical_and on int1 is fast for bool input, but the
# int->int1 conversion is slow for integer/float input.
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config_)
@triton.jit
def logical_and_bool_kernel(x, y):
    return x.to(tl.int1).logical_and(y.to(tl.int1))


# int32 path: tl.where avoids the int->int1 materialization, fast for
# integer/float input but slow for bool (int32->bool store).
@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config_)
@triton.jit
def logical_and_int_kernel(x, y):
    return tl.where((x != 0) & (y != 0), 1, 0)


def logical_and(A, B):
    logger.debug("GEMS_KUNLUNXIN LOGICAL_AND")
    if A.dtype == torch.bool:
        return logical_and_bool_kernel(A, B)
    return logical_and_int_kernel(A, B)


def logical_and_(A, B):
    logger.debug("GEMS_KUNLUNXIN LOGICAL_AND_")
    if A.dtype == torch.bool:
        logical_and_bool_kernel(A, B, out0=A)
    else:
        logical_and_int_kernel(A, B, out0=A)
    return A
