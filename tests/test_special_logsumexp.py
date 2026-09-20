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


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [True, False])
def test_special_logsumexp(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("shape", [(16, 64, 128)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dims", [[0, 1], [0, 2], [1, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_special_logsumexp_multi_dim(shape, dtype, dims, keepdim):
    """Dedicated test for the multi-dim (len(dim) > 1) branch."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=dims, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=dims, keepdim=keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logsumexp_single_element(dtype):
    """Single-element tensor: logsumexp(x) == x."""
    inp = torch.randn((1,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=0)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=0)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logsumexp_large_values(dtype):
    """Numerical stability with very large positive values."""
    # Shape (16, 64) provides enough reduction elements per row to stress
    # the kernel's max-shift numerical stability with large values.
    shape = (16, 64)
    large_val = torch.tensor(1000.0, dtype=dtype)
    inp = torch.full(shape, large_val, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logsumexp_negative_large_values(dtype):
    """Numerical stability with very large negative values mixed with ones."""
    # Shape (8, 32) with one positive per row verifies the max-shift
    # algorithm correctly picks the max and handles near-zero exp terms.
    shape = (8, 32)
    # Most elements are very negative, one positive element per row
    inp = torch.full(shape, -1000.0, dtype=dtype, device=flag_gems.device)
    # Set one positive element per row to verify max-shift algorithm
    inp[:, 0] = 1.0
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logsumexp_all_negative_inf(dtype):
    """Edge case: input tensor where all elements are -inf."""
    # Shape (4, 16) ensures multi-row testing of the all-negative-inf
    # corner case where log(sum(exp(-inf))) must return -inf per row.
    shape = (4, 16)
    inp = torch.full(shape, float("-inf"), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logsumexp_zeros(dtype):
    """Input of all zeros: logsumexp(zeros, dim) = log(N)."""
    # Shape (4, 16) verifies that a uniform zero tensor produces
    # log(N) per row, exercising a common numerical baseline.
    shape = (4, 16)
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=1)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logsumexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logsumexp_extreme_mixed(dtype):
    """Mixed extreme values: large positives, large negatives, and zeros."""
    # Shape (8, 32) with extreme positive/negative per row stresses
    # numerical precision when logsumexp reduces across very wide ranges.
    shape = (8, 32)
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    inp[:, 0] = 1000.0  # Very large positive (float16-safe)
    inp[:, 1] = -1000.0  # Very large negative (float16-safe)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.special.logsumexp(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.special.logsumexp(inp, dim=1)

    utils.gems_assert_close(res_out, ref_out, dtype)
