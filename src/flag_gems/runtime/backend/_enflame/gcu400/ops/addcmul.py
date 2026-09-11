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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    is_tensor=[True, True, True, False], promotion_methods=[(0, 1, 2, "DEFAULT")]
)
@triton.jit
def addcmul_forward(x, t1, t2, value):
    return x + value * t1 * t2


def addcmul(inp, tensor1, tensor2, *, value=1.0, out=None):
    logger.debug("GEMS_ENFLAME ADDCMUL")
    if out is not None:
        broadcast_shape = torch.broadcast_shapes(
            inp.shape, tensor1.shape, tensor2.shape
        )
        if list(out.shape) != list(broadcast_shape):
            out.resize_(broadcast_shape)
        addcmul_forward(inp, tensor1, tensor2, value, out0=out)
        return out
    else:
        return addcmul_forward(inp, tensor1, tensor2, value)
