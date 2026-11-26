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


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
# special.* Chebyshev polynomials: torch ref only supports float32
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t(shape, dtype):
    # x in [-1, 1] (Chebyshev domain)
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x)
    n = 3

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, n)
    res_out = flag_gems.special_chebyshev_polynomial_t(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Test values outside [-1, 1] — the recurrence is valid for all real x
@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_out_of_domain(dtype):
    x_vals = [-2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0]
    x = torch.tensor(x_vals, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    n = 3

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, n)
    res_out = flag_gems.special_chebyshev_polynomial_t(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.parametrize("degree", [0, 1, 2, 5])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_degrees(degree, dtype):
    """T_0 = 1 and T_1 = x are special-cased, so the low degrees are covered."""
    x = torch.rand((64,), dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x)

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, degree)
    res_out = flag_gems.special_chebyshev_polynomial_t(x, degree)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_tensor_degree(dtype):
    """The default overload takes n as a tensor rather than a Python scalar."""
    x = torch.rand((64,), dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x)
    n = torch.tensor(3, dtype=torch.int64, device=flag_gems.device)
    ref_n = utils.to_reference(n)

    ref_out = torch.ops.aten.special_chebyshev_polynomial_t.default(ref_x, ref_n)
    res_out = flag_gems.special_chebyshev_polynomial_t(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t_out
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_out(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x)
    n = 3

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, n)

    out = torch.empty_like(x)
    res_out = flag_gems.ops.special_chebyshev_polynomial_t_out(x, n, out)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    # the out variant must write in place and return that very tensor
    assert res_out.data_ptr() == out.data_ptr()


@pytest.mark.special_chebyshev_polynomial_t
def test_special_chebyshev_polynomial_t_rejects_unsupported_dtype():
    """The kernel is float32/float64 only, so other dtypes must be rejected."""
    x = torch.randint(0, 4, (16,), dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(ValueError):
        flag_gems.special_chebyshev_polynomial_t(x, 3)
