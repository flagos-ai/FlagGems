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

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, "INT_TO_FLOAT")],
    num_outputs=1,
    config=_config,
)
@triton.jit
def dequantize_func(inp, zero_point, scale):
    return (inp - zero_point) * scale


def dequantize(a):
    """Dequantize a quantized tensor (qint8/quint8/qint32) to float32 on XPU."""
    logger.debug("GEMS_KUNLUNXIN DEQUANTIZE")

    scale = float(a.q_scale())
    zero_point = int(a.q_zero_point())

    if a.numel() == 0:
        return torch.empty(a.shape, dtype=torch.float32, device=a.device)

    # `a.int_repr()` has no kernel in the XPU build; use a zero-copy int8 view
    # of the quantized storage instead. The arithmetic stays in the kernel.
    raw = torch.empty(0, dtype=torch.int8, device=a.device)
    raw.set_(a.untyped_storage(), a.storage_offset(), a.size(), a.stride())
    if not raw.is_contiguous():
        raw = raw.contiguous()

    return dequantize_func(raw, zero_point, scale)
