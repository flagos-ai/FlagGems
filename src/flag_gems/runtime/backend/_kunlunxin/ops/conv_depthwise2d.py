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


def _conv_depthwise2d(input, weight, kernel_size, bias, stride, padding, dilation):
    logger.debug("GEMS_KUNLUNXIN CONV_DEPTHWISE2D")
    assert (
        input.ndim == 4
    ), "Invalid input tensor must be 4D, recevied shape {input.shape}"
    assert (
        weight.shape[0] % input.shape[1] == 0
    ), "Output channels must be multiple of input, recevied output {weught.shape[0], input {input.shape[0]}}"
    assert (
        weight.shape[1] == 1
    ), "input channels of per goups must be 1, recevied {weight.shape[1]}"
    groups = input.shape[1]
    return conv2d(input, weight, bias, stride, padding, dilation, groups)
