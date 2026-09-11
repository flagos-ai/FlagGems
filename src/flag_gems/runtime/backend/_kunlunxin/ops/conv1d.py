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
                input.unsqueeze(-1),
                weight.unsqueeze(-1),
                bias,
                (stride_width, 1),
                (padding_width, 0),
                (dilation_width, 1),
                groups,
            ).squeeze(-1)[..., (ol - il) :]
        elif padding == "valid":
            # For "valid" padding, pass the string directly to conv2d
            # conv2d will handle it properly in its own logic
            return conv2d(
                input.unsqueeze(-1),
                weight.unsqueeze(-1),
                bias,
                (stride_width, 1),
                padding,  # Pass string "valid" directly
                (dilation_width, 1),
                groups,
            ).squeeze(-1)
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only 'valid'/'same' are allowed."
            )
    elif isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding
    return conv2d(
        input.unsqueeze(-1),
        weight.unsqueeze(-1),
        bias,
        (stride_width, 1),
        (padding_width, 0),
        (dilation_width, 1),
        groups,
    ).squeeze(-1)
