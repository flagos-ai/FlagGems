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


def thnn_conv2d_impl(
    input, weight, kernel_size=0, bias=None, stride=1, padding=0, groups=1
):
    logger.debug("GEMS_SPACEMIT THNN_CONV2D")
    dilation = 1
    return conv2d(input, weight, bias, padding, stride, dilation, groups)


def thnn_conv2d(input, weight, kernel_size=0, bias=None, stride=1, padding=0, groups=1):
    return thnn_conv2d_impl(input, weight, kernel_size, bias, stride, padding, groups)
