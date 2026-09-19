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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    NONZERO_SHAPES = [(2, 32)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    NONZERO_SHAPES = utils.REDUCTION_SHAPES + [(2637,)]

random.seed(time.time() // 100)


@pytest.mark.nonzero
@pytest.mark.parametrize("shape", NONZERO_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES + [torch.bool])
def test_nonzero(shape, dtype):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=torch.int, device=flag_gems.device).to(
            dtype
        )
    elif dtype in utils.INT_DTYPES:
        inp = torch.randint(-3, 3, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, False)
    ref_out = torch.nonzero(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.nonzero(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.nonzero
@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0), (2, 0, 3)])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
@pytest.mark.parametrize("as_tuple", [False, True])
def test_nonzero_empty(shape, dtype, as_tuple):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.nonzero(inp, as_tuple=as_tuple)
    direct_out = flag_gems.nonzero(inp, as_tuple=as_tuple)
    with flag_gems.use_gems(include=["nonzero", "nonzero_numpy"]):
        dispatched_out = torch.nonzero(inp, as_tuple=as_tuple)
        method_out = inp.nonzero(as_tuple=as_tuple)

    for result in (direct_out, dispatched_out, method_out):
        if as_tuple:
            assert isinstance(result, tuple)
            assert len(result) == inp.ndim
            actual_tensors, expected_tensors = result, ref_out
        else:
            actual_tensors, expected_tensors = (result,), (ref_out,)
        for actual, expected in zip(actual_tensors, expected_tensors):
            assert actual.shape == expected.shape
            assert actual.dtype == torch.int64
            assert actual.device == inp.device
            utils.gems_assert_equal(actual, expected)


@pytest.mark.nonzero
@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0), (2, 0, 3)])
def test_where_empty_condition(shape):
    inp = torch.empty(shape, dtype=torch.bool, device=flag_gems.device)
    ref_out = torch.where(inp)
    with flag_gems.use_gems(include=["nonzero", "nonzero_numpy"]):
        res_out = torch.where(inp)

    assert len(res_out) == inp.ndim
    for actual, expected in zip(res_out, ref_out):
        assert actual.shape == expected.shape
        assert actual.dtype == torch.int64
        assert actual.device == inp.device
        utils.gems_assert_equal(actual, expected)
