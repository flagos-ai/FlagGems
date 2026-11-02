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

from flag_gems.ops.conv1d import conv1d as shared_conv1d
from flag_gems.ops.conv3d import conv3d as shared_conv3d

from .conv2d import (
    Conv2d,
    conv2d_would_convert_input_to_channels_last,
    conv2d_would_use_3x3_fused_cl_full_output,
    conv2d_would_use_tme_3x3s1,
)

logger = logging.getLogger(__name__)


def _conv2d_output_layout(input, out, weight, stride, padding, dilation, groups):
    """Return torch's conv output layout for the cudnn entry point.

    The MUSA conv2d op always allocates channels_last because its kernels
    store in that layout. cudnn_convolution must match torch instead:
    channels_last only when the input is channels_last or when the fast path
    internally converts the input; otherwise a contiguous NCHW tensor.
    """
    if input.is_contiguous(memory_format=torch.channels_last):
        return out
    if (
        conv2d_would_use_tme_3x3s1(input, weight, stride, padding, dilation, groups)
        or conv2d_would_use_3x3_fused_cl_full_output(
            input, weight, stride, padding, dilation, groups
        )
        or conv2d_would_convert_input_to_channels_last(
            input, weight, stride, padding, dilation, groups
        )
    ):
        return out
    return out.contiguous()


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
    """MTHREADS implementation of the bias-free cuDNN convolution entry point.

    ``benchmark``, ``deterministic``, and ``allow_tf32`` match the shared
    Triton wrapper semantics: they are accepted but do not select an algorithm.
    """
    logger.debug("GEMS_MTHREADS CUDNN_CONVOLUTION")

    ndim = input.ndim - 2

    def extract_param(param, expected_len):
        if isinstance(param, (list, tuple)):
            if len(param) == expected_len:
                return param if expected_len > 1 else param[0]
            if len(param) == 1:
                return param[0]
        return param

    if ndim == 1:
        return shared_conv1d(
            input,
            weight,
            bias=None,
            stride=extract_param(stride, 1),
            padding=extract_param(padding, 1),
            dilation=extract_param(dilation, 1),
            groups=groups,
        )
    if ndim == 2:
        # Inlined from _mthreads/ops/conv2d.py::conv2d so this entry point can
        # fix up the output layout without changing the conv2d operator.
        out = Conv2d.apply(input, weight, None, stride, padding, dilation, groups)
        return _conv2d_output_layout(
            input, out, weight, stride, padding, dilation, groups
        )
    if ndim == 3:
        return shared_conv3d(
            input,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
    raise ValueError(
        "cudnn_convolution only supports 1D, 2D, and 3D convolutions, "
        f"got input with {ndim} spatial dimensions"
    )
