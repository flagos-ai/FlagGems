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

    # bfloat16 has no working kernel path on this XPU stack: the vendor
    # conv2d/conv3d handler rejects bf16 launches (xpuLaunchKernel err_code 1/4)
    # and the Triton conv3d kernel aborts inside TritonXPULegalize.  Compute in
    # fp32 and cast back -- the same promotion conv2d/conv3d already apply to
    # fp16 (there for overflow; here for a missing low-precision kernel).
    orig_dtype = input.dtype
    compute_fp32 = orig_dtype == torch.bfloat16
    if compute_fp32:
        input = input.to(torch.float32)
        weight = weight.to(torch.float32)

    if dimensions == 1:
        out = conv1d(input, weight, None, stride, padding, dilation, groups)
    elif dimensions == 2:
        out = conv2d(input, weight, None, stride, padding, dilation, groups)
    else:
        out = conv3d(input, weight, None, stride, padding, dilation, groups)

    if compute_fp32:
        out = out.to(orig_dtype)
    return out
