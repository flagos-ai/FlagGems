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


def replace_zeros(inp):
    return torch.where(inp == 0, 1, inp)


@pytest.mark.remainder
@pytest.mark.remainder_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)

    inp1 = replace_zeros(inp1)
    inp2 = replace_zeros(inp2)

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1 % ref_inp2
    with flag_gems.use_gems():
        res_out = inp1 % inp2

    utils.gems_assert_equal(res_out, ref_out)

    for d in inp2.flatten()[:2]:
        d = d.item()
        ref_out = ref_inp1 % d
        with flag_gems.use_gems():
            res_out = inp1 % d
        utils.gems_assert_equal(res_out, ref_out)

        ref_out = d % ref_inp1
        with flag_gems.use_gems():
            res_out = d % inp1
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder_(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    inp1 = replace_zeros(inp1.clone())
    inp2 = replace_zeros(inp2)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.remainder_(ref_inp2)

    with flag_gems.use_gems():
        res_out = inp1.remainder_(inp2)

    utils.gems_assert_equal(res_out, ref_out)

    ref_inp1 = utils.to_reference(inp1.clone(), False)
    for d in inp2.flatten()[:2]:
        d = d.item()
        ref_out = ref_inp1.remainder_(d)

        with flag_gems.use_gems():
            res_out = inp1.remainder_(d)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("scalar", [7, -7])
def test_remainder_scalar(shape, dtype, scalar):
    inp = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)

    ref_inp = utils.to_reference(inp, False)
    ref_out = torch.remainder(ref_inp, scalar)

    with flag_gems.use_gems():
        res_out = torch.remainder(inp, scalar)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder_scalar_(shape, dtype):
    inp = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    scalar = (
        torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            (1,),
            dtype=dtype,
            device="cpu",
        )
        .to(flag_gems.device)
        .item()
    )

    if scalar == 0:
        scalar = 1

    ref_inp = utils.to_reference(inp.clone(), False)
    ref_out = ref_inp.remainder_(scalar)

    with flag_gems.use_gems():
        res_out = inp.remainder_(scalar)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_scalar_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder_scalar_tensor(shape, dtype):
    inp = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)

    inp = replace_zeros(inp)

    ref_inp = utils.to_reference(inp, False)

    scalar = 7
    ref_out = torch.remainder(torch.tensor(scalar, dtype=dtype), ref_inp)
    with flag_gems.use_gems():
        res_out = torch.remainder(scalar, inp)

    utils.gems_assert_equal(res_out, ref_out)
