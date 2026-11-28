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

"""GCU400-specific conv_depthwise3d.

aten::conv_depthwise3d is depth-wise convolution: every input channel is
convolved with its own single-channel 3-D filter, i.e. a grouped convolution
with ``groups == in_channels`` and a weight of shape ``(C, 1, kD, kH, kW)``.
Because that is exactly the contract of the already-supported grouped path of
this backend's conv3d, the op delegates the whole computation to it: no host
side arithmetic, no aten/CPU compute, every tap is accumulated by the conv3d
Triton kernels.

The generic grouped path of conv3d pads each group's single input channel up to
16 channels (the minimum K of ``tl.dot``) and zero-fills the matching weight
channels, which keeps the result bit-identical to the unpadded depth-wise
convolution while satisfying the dot-product shape constraint.
"""

import logging

from .conv3d import conv3d

logger = logging.getLogger(__name__)


def conv_depthwise3d(input, weight, kernel_size, bias, stride, padding, dilation):
    """Depth-wise 3-D convolution in the aten::conv_depthwise3d sense.

    Args:
        input: (N, C, D, H, W) activation.
        weight: (C, 1, kD, kH, kW) per-channel filter (also encodes
            ``kernel_size``).
        kernel_size: unused, kept for aten signature compatibility; the actual
            kernel extent is ``weight.shape[2:]``.
        bias: optional (C,) bias.
        stride, padding, dilation: int or 3-element sequences along (D, H, W).

    Returns:
        (N, C, out_d, out_h, out_w) tensor in ``input.dtype``.
    """
    num_channels = input.shape[1]
    if weight.shape[0] != num_channels or weight.shape[1] != 1:
        raise ValueError(
            "conv_depthwise3d expects weight of shape (C, 1, kD, kH, kW) "
            f"matching the {num_channels} input channels, got {tuple(weight.shape)}"
        )

    # groups == in_channels turns the grouped conv3d into a depth-wise conv.
    return conv3d(input, weight, bias, stride, padding, dilation, groups=num_channels)
