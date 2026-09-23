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


@pytest.mark.logical_xor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES,
)
def test_logical_xor_(shape, dtype):
    if dtype in utils.ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in utils.BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )

    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.logical_xor_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.logical_xor_(inp2)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp1, ref_inp1)


@pytest.mark.logical_xor_
def test_logical_xor_bool_scalar():
    # 2D shape for scalar-vs-tensor inplace xor validation
    shape = (128, 128)
    inp = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    scalar = torch.tensor(True, dtype=torch.bool, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone())
    ref_scalar = utils.to_reference(scalar)
    ref_out = ref_inp.logical_xor_(ref_scalar)
    with flag_gems.use_gems():
        res_out = inp.logical_xor_(scalar)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.logical_xor_
def test_logical_xor_broadcast():
    # 2D shapes with broadcast along dim 1
    shape = (64, 128)
    broad_shape = (64, 1)
    inp = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    other = torch.randint(0, 2, broad_shape, dtype=torch.bool, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)
    ref_out = ref_inp.logical_xor_(ref_other)
    with flag_gems.use_gems():
        res_out = inp.logical_xor_(other)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.logical_xor_
def test_logical_xor_noncontiguous():
    # 2D shape where transpose makes tensors non-contiguous
    shape = (128, 64)
    inp = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    other = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)

    inp_t = inp.transpose(0, 1)
    other_t = other.transpose(0, 1)

    ref_inp = utils.to_reference(inp_t.clone())
    ref_out = ref_inp.logical_xor_(utils.to_reference(other_t))
    with flag_gems.use_gems():
        res_out = inp_t.logical_xor_(other_t)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp_t, ref_inp)
