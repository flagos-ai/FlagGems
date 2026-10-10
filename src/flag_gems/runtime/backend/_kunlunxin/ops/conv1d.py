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
import math

from .conv2d import conv2d

logger = logging.getLogger(__name__)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    logger.debug("GEMS_KUNLUNXIN CONV1D")
    if isinstance(stride, (list, tuple)):
        stride_width = stride[0]
    else:
        stride_width = stride

    if isinstance(dilation, (list, tuple)):
        dilation_width = dilation[0]
    else:
        dilation_width = dilation

    # The 1-D length axis is mapped onto the W axis of a 2-D convolution with
    # H == 1, NOT onto H with W == 1.  Both are pure views of the same memory,
    # but the vendor conv2d_fusion kernel is ~3.5x faster with H == 1: W is
    # the contiguous axis, so a W == 1 input gives every program a one-element
    # row.  Measured on (32,64,512)x(64,64,3) fp32: 74 us (W == 1) versus
    # 21 us (H == 1).  The downstream 2-D helpers all take an (N,C,H,W)
    # tensor, so this is the only change needed.
    if isinstance(padding, str):
        if padding == "same":
            assert stride == 1, (
                f"Doesn't support any stride values other than 1 in padding = 'same' mode, "
                f"received stride value {stride}"
            )
            il = input.shape[-1]
            kernel_size = weight.shape[-1]
            padding_width = math.ceil(
                (stride_width * (il - 1) + 1 + dilation_width * (kernel_size - 1) - il)
                / 2
            )
            ol = int(
                (il + 2 * padding_width - dilation_width * (kernel_size - 1) - 1)
                / stride_width
                + 1
            )
            return conv2d(
                input.unsqueeze(2),
                weight.unsqueeze(2),
                bias,
                (1, stride_width),
                (0, padding_width),
                (1, dilation_width),
                groups,
            ).squeeze(2)[..., (ol - il) :]
        elif padding == "valid":
            # For "valid" padding, pass the string directly to conv2d
            # conv2d will handle it properly in its own logic
            return conv2d(
                input.unsqueeze(2),
                weight.unsqueeze(2),
                bias,
                (1, stride_width),
                padding,  # Pass string "valid" directly
                (1, dilation_width),
                groups,
            ).squeeze(2)
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only 'valid'/'same' are allowed."
            )
    elif isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding
    return conv2d(
        input.unsqueeze(2),
        weight.unsqueeze(2),
        bias,
        (1, stride_width),
        (0, padding_width),
        (1, dilation_width),
        groups,
    ).squeeze(2)
