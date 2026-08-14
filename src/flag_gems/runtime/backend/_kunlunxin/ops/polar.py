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
from ._fp64_compat import empty_fp64

logger = logging.getLogger(__name__)


@pointwise_dynamic(
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def polar_kernel(abs, angle):
    # XPU libdevice does not provide f64 sin/cos.  Keep f64 storage metadata,
    # but evaluate the operation in the backend's supported compute dtype.
    abs_f32 = abs.to(tl.float32)
    angle_f32 = angle.to(tl.float32)
    real = abs_f32 * tl.cos(angle_f32)
    imag = abs_f32 * tl.sin(angle_f32)
    return real, imag


def polar(abs, angle):
    logger.debug("GEMS_KUNLUNXIN POLAR")
    if abs.dtype == torch.float64:
        output = empty_fp64((*abs.shape, 2), device=abs.device)
    else:
        output = torch.empty((*abs.shape, 2), dtype=abs.dtype, device=abs.device)

    polar_kernel(abs, angle, out0=output[..., 0], out1=output[..., 1])

    return torch.view_as_complex(output)
