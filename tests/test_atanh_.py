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


@pytest.mark.atanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8 - 0.9
    ref_inp = utils.to_reference(inp, True)

    ref_out = ref_inp.atanh_()

    inp1 = inp.clone()
    res_out = flag_gems.atanh_(inp1)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp1, ref_inp, dtype)
    assert res_out is inp1


# ---------------------------------------------------------------------------
# Domain-matrix additions: value-range coverage ([-1,1], [0,1], [-1,0] and the
# out-of-domain edges beyond both bounds) crossed with shapes and dtypes.
# ---------------------------------------------------------------------------

ATANH_INPLACE_VALUE_RANGES = [
    "full_domain",
    "positive",
    "negative",
    "above_max",
    "below_min",
]


def _atanh__range_input(shape, dtype, range_name):
    """Build inputs for the atanh_ domain matrix: valid domain (-1, 1) plus the
    out-of-domain bands beyond both endpoints (ATen yields NaN there)."""
    if range_name == "full_domain":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -1.0, 1.0
        )
    if range_name == "positive":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            0.0, 1.0
        )
    if range_name == "negative":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -1.0, 0.0
        )
    if range_name == "above_max":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            1.5, 4.0
        )
    if range_name == "below_min":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -4.0, -1.5
        )
    raise ValueError(f"Unknown range_name: {range_name}")


@pytest.mark.atanh_
@pytest.mark.parametrize("range_name", ATANH_INPLACE_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh__range(shape, dtype, range_name):
    inp = _atanh__range_input(shape, dtype, range_name)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = ref_inp.atanh_()

    inp1 = inp.clone()
    res_out = flag_gems.atanh_(inp1)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(inp1, ref_inp, dtype, equal_nan=True)
    assert res_out is inp1


@pytest.mark.atanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh__inplace_identity(shape, dtype):
    # In-place contract: the returned tensor is the input object itself.
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8 - 0.9
    res_out = flag_gems.atanh_(inp)
    assert res_out is inp


@pytest.mark.atanh_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh__special_values(dtype):
    # Param coverage type 5: zero (with sign preserved), nan and the infinities.
    values = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values.clone(), True)

    ref_out = ref_values.atanh_()

    res_values = values.clone()
    res_out = flag_gems.atanh_(res_values)

    utils.gems_assert_close(res_out, ref_out.to(dtype), dtype, equal_nan=True)
    assert res_out is res_values


@pytest.mark.atanh_
@pytest.mark.parametrize("shape", utils.POINTWISE_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh__noncontiguous(shape, dtype):
    # Param coverage type 5: non-contiguous inputs must produce the same
    # values as their contiguous copies. Shapes with any 1-sized dim give
    # a contiguous result ( stride-1 squeeze), and 0-dim has no stride at
    # all, so both are excluded from the shape list.
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8 - 0.9
    noncontig = inp.repeat(2, *[1] * (inp.ndim - 1))[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous().clone(), True)

    ref_out = ref_inp.atanh_()
    res_out = flag_gems.atanh_(noncontig)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.atanh_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atanh__zero_size(dtype):
    # Param coverage type 5: 0-size tensors must pass through unchanged.
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 1.8 - 0.9
        res_out = flag_gems.atanh_(inp)
        assert res_out.numel() == 0
        assert res_out.dtype == dtype
