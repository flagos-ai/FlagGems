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
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    # torch reference only supports float32 on CUDA
    [torch.float32],
)
def test_accuracy_special_chebyshev_polynomial_t(shape, dtype):
    # T_n is defined for every real x, but the polynomial grows quickly outside
    # [-1, 1], so x is clamped to the interval where the reference is well
    # conditioned. The out-of-interval behaviour is covered by the test below.
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device).clamp(-0.99, 0.99)
    # n is the degree of the polynomial, use small positive integers
    n = torch.randint(0, 5, shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_n = utils.to_reference(n, True)

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, ref_n)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_t(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_accuracy_special_chebyshev_polynomial_t_scalar_order(n):
    # A python int for n exercises the scalar branch of the wrapper, which wraps
    # the value into a tensor before launching the kernel.
    dtype = torch.float32
    x = torch.randn((64, 64), dtype=dtype, device=flag_gems.device).clamp(-0.99, 0.99)
    ref_x = utils.to_reference(x, True)

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, n)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_t(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_chebyshev_polynomial_t
def test_accuracy_special_chebyshev_polynomial_t_boundary_values():
    # T_n(1) = 1 and T_n(-1) = (-1)^n for every n, so the interval endpoints are
    # a useful exact check that the polynomial selection is wired correctly.
    dtype = torch.float32
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=dtype, device=flag_gems.device)
    n = torch.tensor([0, 1, 2, 3, 4], dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_n = utils.to_reference(n, True)

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, ref_n)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_t(x, n)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_chebyshev_polynomial_t
def test_accuracy_special_chebyshev_polynomial_t_non_contiguous():
    # A transposed view exercises the strided path of the pointwise kernel.
    dtype = torch.float32
    x = (
        torch.randn((16, 32), dtype=dtype, device=flag_gems.device)
        .clamp(-0.99, 0.99)
        .t()
    )
    n = torch.randint(0, 5, (16, 32), dtype=dtype, device=flag_gems.device).t()

    ref_x = utils.to_reference(x, True)
    ref_n = utils.to_reference(n, True)

    ref_out = torch.special.chebyshev_polynomial_t(ref_x, ref_n)
    with flag_gems.use_gems():
        res_out = torch.special.chebyshev_polynomial_t(x, n)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_chebyshev_polynomial_t
@pytest.mark.parametrize("n", [-1, 6])
def test_accuracy_special_chebyshev_polynomial_t_unsupported_order(n):
    # Only n in [0, 5] has an explicit formula in the kernel, so orders outside
    # that range are rejected instead of returning a wrong result silently.
    x = torch.randn((32,), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(ValueError):
        with flag_gems.use_gems():
            torch.special.chebyshev_polynomial_t(x, n)
