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


@pytest.mark.exp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.exp(ref_inp)
    res_out = flag_gems.exp(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exp_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.exp_(ref_inp)
    res_out = flag_gems.exp_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exp_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.exp(ref_inp, out=ref_out)
    res_out = torch.empty_like(inp)
    flag_gems.exp_out(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp
def test_exp_float64_precision():
    # FP64 must not be downcast to FP32 during computation.
    ref_inp = torch.tensor(
        [0.3, 90.0, 700.0],
        dtype=torch.float64,
    )
    inp = ref_inp.to(flag_gems.device)

    expected = torch.exp(ref_inp)

    with flag_gems.use_gems():
        actual = torch.exp(inp)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(
        actual.cpu(), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.exp
def test_exp_float64_gradcheck():
    inp = torch.tensor(
        [[0.1, 0.3, 0.7], [1.0, 2.0, 10.0]],
        dtype=torch.float64,
        device=flag_gems.device,
        requires_grad=True,
    )

    with flag_gems.use_gems():
        assert torch.autograd.gradcheck(torch.exp, (inp,))
