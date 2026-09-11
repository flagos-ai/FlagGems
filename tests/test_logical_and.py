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


@pytest.mark.logical_and
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES,
)
def test_logical_and(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

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
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.logical_and(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.logical_and(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.logical_and_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES,
)
def test_logical_and_(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

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

    ref_out = ref_inp1.logical_and_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.logical_and_(inp2)

    utils.gems_assert_equal(res_out, ref_out)
