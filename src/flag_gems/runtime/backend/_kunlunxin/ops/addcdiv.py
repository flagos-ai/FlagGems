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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# addcdiv is a memory-bound elementwise op (reads 3 tensors, writes 1) whose
# only compute is a division. The default pointwise_dynamic codegen emits a tiny
# 256-element tile with no unrolling, which underutilizes the XPU badly
# (~250x slower than torch). Use the same tuned config as div.py (the sibling
# division op): larger buffer, unrolling, and closed vectorization measure best
# on XPU for this op (x0.99 vs torch, vs x0.87 default / x0.74 with vec open).
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, True, False],
    promotion_methods=[(0, 1, 2, "DEFAULT")],
    config=config_,
)
@triton.jit
def addcdiv_kernel(x, t1, t2, value):
    return x + value * (t1 / t2)


def addcdiv(inp, tensor1, tensor2, value=1.0, out=None):
    logger.debug("GEMS_KUNLUNXIN ADDCDIV")

    if out is None:
        out = torch.empty_like(inp)

    addcdiv_kernel(inp, tensor1, tensor2, value, out0=out)

    return out


def addcdiv_out(inp, tensor1, tensor2, *, value=1.0, out):
    logger.debug("GEMS_KUNLUNXIN ADDCDIV_OUT")
    addcdiv_kernel(inp, tensor1, tensor2, value, out0=out)
    return out


def addcdiv_(inp, tensor1, tensor2, value=1.0):
    logger.debug("GEMS_KUNLUNXIN ADDCDIV_")
    addcdiv_kernel(inp, tensor1, tensor2, value, out0=inp)
    return inp
