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


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.hermite_polynomial_h reference only supports float32 and float64
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h(shape, dtype):
    # Test with tensor n in [0, 9]
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: Not supported")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 10, (1,), device=flag_gems.device).squeeze()

    # Compare against PyTorch in the same dtype: gems replicates PyTorch's
    # in-dtype recurrence, so no float64 upcast is used for the reference.
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.hermite_polynomial_h(ref_inp, utils.to_reference(n))
    res_out = flag_gems.special_hermite_polynomial_h(inp, n)

    # gems uses the same in-dtype recurrence as PyTorch, so results match the
    # reference to default precision for both float32 and float64.
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# special.hermite_polynomial_h reference only supports float32 and float64
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h_scalar(shape, dtype):
    # Test with scalar n = 9 (largest supported degree)
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: Not supported")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = 9

    # Compare against PyTorch in the same dtype (no float64 upcast).
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.hermite_polynomial_h(ref_inp, n)
    res_out = flag_gems.special_hermite_polynomial_h(inp, n)

    # gems uses the same in-dtype recurrence as PyTorch, so results match the
    # reference to default precision for both float32 and float64.
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_hermite_polynomial_h
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_hermite_polynomial_h_out_of_range(dtype):
    # Verify that n >= 10 or n < 0 raises ValueError
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: Not supported")
    inp = torch.randn(4, 4, dtype=dtype, device=flag_gems.device)

    with pytest.raises(ValueError, match="only supports n"):
        flag_gems.special_hermite_polynomial_h(inp, 10)
    with pytest.raises(ValueError, match="only supports n"):
        flag_gems.special_hermite_polynomial_h(inp, -1)

    # Verify that tensor n with values >= 10 raises ValueError
    with pytest.raises(ValueError, match="only supports n"):
        n_bad = torch.tensor(10, dtype=torch.int32, device=flag_gems.device)
        flag_gems.special_hermite_polynomial_h(inp, n_bad)
    with pytest.raises(ValueError, match="only supports n"):
        n_bad = torch.tensor(-1, dtype=torch.int32, device=flag_gems.device)
        flag_gems.special_hermite_polynomial_h(inp, n_bad)
