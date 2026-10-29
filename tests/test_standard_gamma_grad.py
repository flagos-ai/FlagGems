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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


_REGIMES = [
    pytest.param((0.5, 4.5), (0.5, 4.5), id="general"),
    pytest.param((10.0, 50.0), (0.9, 1.1), id="large-near-mode"),
    pytest.param((10.0, 50.0), (0.5, 0.85), id="large-below-mode"),
    pytest.param((10.0, 50.0), (1.15, 1.5), id="large-above-mode"),
    pytest.param((0.5, 4.5), (0.01, 0.71), id="small-output"),
]


@pytest.mark.standard_gamma_grad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("alpha_range,output_range", _REGIMES)
def test_standard_gamma_grad_regimes(shape, dtype, alpha_range, output_range):
    torch.manual_seed(0)
    alpha_lo, alpha_hi = alpha_range
    out_lo, out_hi = output_range
    alpha = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) * (alpha_hi - alpha_lo)
        + alpha_lo
    )
    if alpha_lo >= 10.0:
        output = alpha * (
            torch.rand(shape, dtype=dtype, device=flag_gems.device) * (out_hi - out_lo)
            + out_lo
        )
    else:
        output = (
            torch.rand(shape, dtype=dtype, device=flag_gems.device) * (out_hi - out_lo)
            + out_lo
        )
    alpha_before = utils.to_reference(alpha.clone())
    output_before = utils.to_reference(output.clone())
    ref_alpha = utils.to_reference(alpha, True)
    ref_output = utils.to_reference(output, True)

    reference = torch._standard_gamma_grad(ref_alpha, ref_output)
    result = flag_gems.standard_gamma_grad(alpha, output)

    assert result.device == alpha.device
    assert result.shape == shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype)
    utils.gems_assert_equal(alpha, alpha_before)
    utils.gems_assert_equal(output, output_before)


@pytest.mark.standard_gamma_grad
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_standard_gamma_grad_noncontiguous(dtype):
    torch.manual_seed(0)
    alpha = (torch.rand((7, 5), dtype=dtype, device=flag_gems.device) + 0.5).T
    output = (torch.rand((7, 5), dtype=dtype, device=flag_gems.device) + 0.1).T
    assert not alpha.is_contiguous()
    ref_alpha = utils.to_reference(alpha, True)
    ref_output = utils.to_reference(output, True)

    reference = torch._standard_gamma_grad(ref_alpha, ref_output)
    result = flag_gems.standard_gamma_grad(alpha, output)

    assert result.device == alpha.device
    assert result.dtype == alpha.dtype
    assert result.shape == alpha.shape
    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.standard_gamma_grad
def test_standard_gamma_grad_rejects_mismatched_dtype():
    alpha = torch.rand((8,), dtype=torch.float32, device=flag_gems.device) + 0.5
    output = torch.rand((8,), dtype=torch.float64, device=flag_gems.device) + 0.5

    with pytest.raises(RuntimeError):
        flag_gems.standard_gamma_grad(alpha, output)


@pytest.mark.standard_gamma_grad
def test_standard_gamma_grad_rejects_integer_inputs():
    alpha = torch.ones((8,), dtype=torch.int64, device=flag_gems.device)
    output = torch.ones((8,), dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.standard_gamma_grad(alpha, output)


@pytest.mark.standard_gamma_grad
def test_standard_gamma_grad_rejects_nonbroadcastable_shape():
    alpha = torch.ones((2, 3), device=flag_gems.device)
    output = torch.ones((4,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.standard_gamma_grad(alpha, output)
