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


# special.* Chebyshev polynomials only support float32 in PyTorch reference
@pytest.mark.special_chebyshev_polynomial_w
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
# special.* Chebyshev polynomials: torch ref only supports float32
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_w(shape, dtype):
    # x in [-1, 1] (Chebyshev domain)
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x)
    n = 3

    ref_out = torch.special.chebyshev_polynomial_w(ref_x, n)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_w(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Test values outside [-1, 1] — recurrence is valid for all real x
@pytest.mark.special_chebyshev_polynomial_w
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_w_out_of_domain(dtype):
    # Test with |x| > 1 values; W_3(2.0) = 71.0 in PyTorch reference
    x_vals = [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0]
    x = torch.tensor(x_vals, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    n = 3

    ref_out = torch.special.chebyshev_polynomial_w(ref_x, n)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_w(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_w
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_w_out(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x)
    n = 3

    ref_out = torch.special.chebyshev_polynomial_w(ref_x, n)
    with flag_gems.use_gems():
        out = torch.empty_like(x)
        res_out = torch.special.chebyshev_polynomial_w(x, n, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    # Verify output is the same tensor as `out`
    assert res_out.data_ptr() == out.data_ptr()
