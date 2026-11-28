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

from .conv1d import conv1d
from .conv2d import conv2d
from .conv3d import conv3d

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

    orig_dtype = input.dtype
    # bfloat16 runs natively on the 2D and 3D paths. The 1D path, a strided 2D
    # convolution, and geometries without a working kernel are computed in
    # float32 and cast back.
    up_front = orig_dtype == torch.bfloat16 and (
        dimensions == 1 or (dimensions == 2 and (stride[0] != 1 or stride[1] != 1))
    )
    retry_fp32 = orig_dtype == torch.bfloat16 and not up_front
    if up_front:
        input = input.to(torch.float32)
        weight = weight.to(torch.float32)
    while True:
        try:
            if dimensions == 1:
                out = conv1d(input, weight, None, stride, padding, dilation, groups)
            elif dimensions == 2:
                out = conv2d(input, weight, None, stride, padding, dilation, groups)
            else:
                out = conv3d(input, weight, None, stride, padding, dilation, groups)
            break
        except RuntimeError:
            if not retry_fp32:
                raise
            retry_fp32 = False
            input = input.to(torch.float32)
            weight = weight.to(torch.float32)

    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out
