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


@pytest.mark.arcsin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arcsin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arcsin(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arcsin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.arcsin_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)
    utils.gems_assert_close(inp, ref_out, dtype, True)


@pytest.mark.arcsin_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty_like(ref_inp)
    torch.arcsin(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.arcsin(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_boundaries_and_out_of_domain(dtype):
    values = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0, -1.5, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values)

    ref_out = torch.arcsin(ref_values)
    with flag_gems.use_gems():
        res_out = torch.arcsin(values)

    utils.gems_assert_close(res_out, ref_out, dtype, True)
    assert torch.isnan(res_out[-2:]).all()


@pytest.mark.arcsin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_empty_and_scalar(dtype):
    empty = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    scalar = torch.tensor(0.5, dtype=dtype, device=flag_gems.device)

    ref_empty = utils.to_reference(empty)
    ref_scalar = utils.to_reference(scalar)

    with flag_gems.use_gems():
        res_empty = torch.arcsin(empty)
        res_scalar = torch.arcsin(scalar)

    utils.gems_assert_close(res_empty, torch.arcsin(ref_empty), dtype, True)
    utils.gems_assert_close(res_scalar, torch.arcsin(ref_scalar), dtype, True)
