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
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, "DEFAULT")],
    config=config_,
)
@triton.jit
def unscale_func(value, inv_scale):
    return (value.to(tl.float32) * inv_scale).to(value.dtype)


def _amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale):
    logger.debug("GEMS_KUNLUNXIN AMP_FOREACH_NON_FINITE_CHECK_AND_UNSCALE")
    scale = inv_scale.item()
    has_non_finite = False
    for tensor in tensors:
        unscale_func(tensor, scale, out0=tensor)
        if not torch.isfinite(tensor).all():
            has_non_finite = True

    if has_non_finite:
        found_inf.fill_(1.0)
