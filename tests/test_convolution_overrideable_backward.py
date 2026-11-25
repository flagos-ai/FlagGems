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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Backward cases cover direct and transposed convolution, 1D/2D/3D, groups,
# stride, padding and dilation. Channels stay >= 16 (plus contractions of at
# least 16) so the FlagGems im2col kernels have enough K for tl.dot; small
# channels are exercised separately below at a coarser tolerance.
CONV_BACKWARD_CASES = [
    # (input shape, weight shape, stride, padding, dilation, transposed, groups)
    ((2, 16, 10, 10), (16, 16, 3, 3), (1, 1), (1, 1), (1, 1), False, 1),
    ((2, 16, 12, 12), (16, 8, 3, 3), (2, 2), (2, 2), (2, 2), False, 2),
    ((2, 16, 9, 9), (8, 16, 3, 3), (1, 2), (1, 1), (1, 2), False, 1),
    ((2, 16, 11), (16, 16, 3), (1,), (1,), (1,), False, 1),
    ((2, 16, 12), (16, 8, 3), (2,), (1,), (2,), False, 2),
    ((2, 16, 6, 6, 6), (16, 16, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), False, 1),
    # Transposed forward: weight is (in_channels, out_channels/groups, *kernel).
    ((2, 16, 8, 8), (16, 16, 3, 3), (2, 1), (1, 1), (1, 1), True, 1),
    ((2, 16, 8, 8), (16, 8, 3, 3), (1, 1), (2, 0), (2, 1), True, 2),
    ((2, 16, 8), (16, 16, 3), (2,), (1,), (1,), True, 1),
    ((2, 16, 5, 5, 5), (16, 16, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), True, 1),
]


def _forward_reference(
    inp, weight, stride, padding, dilation, transposed, output_padding, groups
):
    return torch.ops.aten.convolution(
        inp,
        weight,
        None,
        list(stride),
        list(padding),
        list(dilation),
        transposed,
        list(output_padding),
        groups,
    )


def _backward_reduction_dim(weight_shape, transposed, groups):
    # Direct conv contracts over in_channels/groups * k...; transposed over
    # in_channels * k... (the weight's leading axis).
    if transposed:
        reduce_dim = weight_shape[0]
    else:
        reduce_dim = weight_shape[1] * groups
    for size in weight_shape[2:]:
        reduce_dim *= size
    return max(reduce_dim, 1)


@pytest.mark.convolution_overrideable_backward
@pytest.mark.parametrize(
    "shape, kernel, stride, padding, dilation, transposed, groups",
    CONV_BACKWARD_CASES,
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_convolution_overrideable_backward(
    shape, kernel, stride, padding, dilation, transposed, groups, dtype
):
    torch.manual_seed(0)
    torch.backends.cudnn.allow_tf32 = False
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    output_padding = [0] * (len(shape) - 2)

    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = _forward_reference(
        ref_inp,
        ref_weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    out_grad = torch.randn(ref_out.shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    res_in_grad = flag_gems.convolution_overrideable_backward(
        out_grad,
        inp,
        weight,
        list(stride),
        list(padding),
        list(dilation),
        transposed,
        output_padding,
        groups,
    )

    assert res_in_grad.shape == inp.shape
    assert res_in_grad.dtype == dtype
    utils.gems_assert_close(
        res_in_grad,
        ref_in_grad,
        dtype,
        reduce_dim=_backward_reduction_dim(kernel, transposed, groups),
    )
