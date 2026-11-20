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

from flag_gems.ops.convolution_backward import (
    _bounded_input_gradient,
    _convolution_backward_impl,
    _convolution_backward_out_impl,
)

logger = logging.getLogger(__name__)


def convolution_backward(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS_ASCEND CONVOLUTION_BACKWARD")
    return _convolution_backward_impl(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
        _input_gradient_config=_bounded_input_gradient,
    )


def convolution_backward_out(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    *,
    out0,
    out1,
    out2,
):
    logger.debug("GEMS_ASCEND CONVOLUTION_BACKWARD_OUT")
    return _convolution_backward_out_impl(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
        out0=out0,
        out1=out1,
        out2=out2,
        _input_gradient_config=_bounded_input_gradient,
    )
