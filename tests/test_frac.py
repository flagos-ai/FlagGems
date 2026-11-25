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


@pytest.mark.frac
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_frac(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.frac(ref_inp)
    res_out = flag_gems.frac(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.frac_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["scalar", "empty", "contiguous", "transposed"])
def test_frac_backward(dtype, layout):
    if layout == "scalar":
        inp = torch.tensor(-2.0, dtype=dtype, device=flag_gems.device)
    elif layout == "empty":
        inp = torch.empty((0, 3), dtype=dtype, device=flag_gems.device)
    else:
        # Exercise both signs, zero and integer inputs (ATen uses identity even
        # at discontinuities), as well as non-unit upstream gradients.
        values = torch.tensor(
            [[-2.0, -1.25, -0.0], [0.0, 1.5, 3.0]],
            dtype=dtype,
            device=flag_gems.device,
        )
        inp = values if layout == "contiguous" else values.T
    grad = torch.randn(inp.shape, dtype=dtype, device=flag_gems.device)
    if layout == "transposed":
        grad = grad.T.contiguous().T

    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    actual = inp.detach().requires_grad_()
    expected = torch.autograd.grad(
        torch.frac(ref_inp), ref_inp, utils.to_reference(grad, False)
    )[0]
    out = flag_gems.frac(actual)
    assert "FracFunctionBackward" in type(out.grad_fn).__name__
    result = torch.autograd.grad(out, actual, grad)[0]
    utils.gems_assert_equal(result, expected)
    utils.gems_assert_equal(result, utils.to_reference(grad))


@pytest.mark.frac_backward
def test_frac_backward_dispatch_registration():
    entries = [entry for entry in flag_gems._FULL_CONFIG if entry[0] == "frac"]
    assert len(entries) == 1
    assert (
        flag_gems.AUTOGRAD_DISPATCH_KEY,
        torch.library.fallthrough_kernel,
    ) in entries[0][3]
