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
from .accuracy_utils import FLOAT_DTYPES, gems_assert_equal

PAD_BACKWARD_CASES = [
    # (input shape, pad, value)
    ((2, 3, 5, 7), [1, 2, 3, 1, 0, 2], 1.5),
    # Negative entries shrink a side; the backward re-grows it.
    ((2, 3, 5, 7), [1, -2, 3, 0, -1, 2], -0.5),
    ((4, 6), [2, 1], 0.0),
    ((3, 4, 5), [0, 0, 2, 1], 7.0),
    # Zero padding: the gradient is the identity.
    ((2, 3, 4), [0, 0, 0, 0], 1.0),
    # Padding every spatial dim; leading dims default to no padding.
    ((2, 3), [1, 2, 0, 0], 3.0),
    # Trailing dims untouched: only the last dimension is padded.
    ((2, 3, 5), [0, 0, 2, 1], 1.0),
    # Rank-1 with a single pad pair.
    ((7,), [2, 3], 0.0),
]


@pytest.mark.constant_pad_nd_backward
@pytest.mark.parametrize("shape,pad,value", PAD_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd_backward(shape, pad, value, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    out_grad = torch.randn(ref_out.shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    # The gradient w.r.t. `value` is a sum, but `value` is a scalar, so only
    # the input gradient is produced here.
    res_in_grad = flag_gems.constant_pad_nd_backward(out_grad, inp, pad, value)

    assert res_in_grad.shape == inp.shape
    assert res_in_grad.dtype == dtype
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.constant_pad_nd_backward
@pytest.mark.parametrize(
    "shape,pad,value",
    [case for case in PAD_BACKWARD_CASES if all(p >= 0 for p in case[1])],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd_backward_reference_grad_values(shape, pad, value, dtype):
    """With non-negative pads the backward is exactly the interior slice."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_shape = torch.constant_pad_nd(inp, pad, value).shape
    out_grad = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)

    res_in_grad = flag_gems.constant_pad_nd_backward(out_grad, inp, pad, value)

    # Pads are listed from the last dimension backwards.
    slices = []
    for dim in range(inp.ndim):
        pad_idx = (inp.ndim - 1 - dim) * 2
        if pad_idx + 1 < len(pad):
            left = pad[pad_idx]
            slices.append(slice(left, left + inp.shape[dim]))
        else:
            slices.append(slice(0, inp.shape[dim]))
    expected = out_grad[(slice(None),) * (out_grad.ndim - inp.ndim) + tuple(slices)]
    assert res_in_grad.shape == expected.shape
    gems_assert_equal(res_in_grad, expected)
