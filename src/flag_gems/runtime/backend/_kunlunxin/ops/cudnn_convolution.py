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

from .conv1d import conv1d
from .conv2d import conv2d

logger = logging.getLogger(__name__)


def _spatial_tuple(value, dimensions, name):
    if isinstance(value, int):
        return (value,) * dimensions
    if isinstance(value, (list, tuple)) and len(value) == dimensions:
        return tuple(value)
    raise ValueError(f"{name} must have {dimensions} values, got {value}")


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """CUDNN-compatible no-bias convolution using native Kunlunxin kernels."""
    logger.debug("GEMS_KUNLUNXIN CUDNN_CONVOLUTION")
    dimensions = input.ndim - 2
    if dimensions not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution expects a 3D, 4D, or 5D input, got {input.ndim}D"
        )

    padding = _spatial_tuple(padding, dimensions, "padding")
    stride = _spatial_tuple(stride, dimensions, "stride")
    dilation = _spatial_tuple(dilation, dimensions, "dilation")
    if dimensions == 1:
        return conv1d(input, weight, None, stride, padding, dilation, groups)
    if dimensions == 2:
        return conv2d(input, weight, None, stride, padding, dilation, groups)
    raise NotImplementedError(
        "Kunlunxin cudnn_convolution does not support 3D inputs: the local conv3d "
        "implementation is not device-resident"
    )
