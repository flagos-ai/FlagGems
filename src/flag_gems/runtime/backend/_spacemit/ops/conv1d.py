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

from .conv2d import conv2d

logger = logging.getLogger(__name__)


def conv1d(input, weight, bias=None, padding=0, stride=1, dilation=1, groups=1):
    logger.debug("GEMS_SPACEMIT CONV1D")

    if isinstance(stride, (list, tuple)):
        stride_width = stride[0]
    else:
        stride_width = stride

    if isinstance(padding, (list, tuple)):
        padding_width = padding[0]
    else:
        padding_width = padding
    return conv2d(
        input.unsqueeze(-1),
        weight.unsqueeze(-1),
        bias,
        (padding_width, 0),
        (stride_width, 1),
        dilation,
        groups,
    ).squeeze(-1)
