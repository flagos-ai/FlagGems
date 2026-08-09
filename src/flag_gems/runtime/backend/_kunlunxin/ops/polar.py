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


@pointwise_dynamic(
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
)
@triton.jit
def polar_kernel(abs, angle):
    real = abs * tl.cos(angle)
    imag = abs * tl.sin(angle)
    return real, imag


def polar(abs, angle):
    logger.debug("GEMS_KUNLUNXIN POLAR")
    # XPU note: writing the two components directly into the interleaved
    # complex layout (stride-2 strided outputs) takes the slow rank-2
    # scalarized codegen path (~40s for 16M elts).  Write two contiguous
    # real/imag buffers on the fast rank-1 path, then interleave them with
    # one coalesced device-side copy (torch.complex) before returning.
    real = torch.empty_like(abs)
    imag = torch.empty_like(abs)

    polar_kernel(abs, angle, out0=real, out1=imag)

    return torch.complex(real, imag)
