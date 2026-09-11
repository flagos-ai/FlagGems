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
from . import conftest as cfg

if cfg.QUICK_MODE:
    REPEAT_INTERLEAVE_SHAPES = [
        (1024, 1024),
        (20, 320, 15),
    ]
    REPEAT_INTERLEAVE_DIM = [-1, 0]
else:
    REPEAT_INTERLEAVE_SHAPES = [
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ]
    REPEAT_INTERLEAVE_DIM = [-1, 0, None]


@pytest.mark.repeat_interleave_self_int
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES + [(1,)])
@pytest.mark.parametrize("dim", REPEAT_INTERLEAVE_DIM)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_repeat_interleave_self_int(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    repeats = 2
    ref_inp = utils.to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave_self_int
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@pytest.mark.parametrize("dim", REPEAT_INTERLEAVE_DIM)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_repeat_interleave_self_int_non_contiguous(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)[::2]
    repeats = 2
    ref_inp = utils.to_reference(inp)

    ref_out = torch.repeat_interleave(ref_inp, repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave_tensor
@pytest.mark.parametrize("shape", utils.UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", [torch.int32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_repeat_interleave_tensor(shape, dtype):
    repeats = torch.randint(0, 30, shape, dtype=dtype, device=flag_gems.device)
    ref_repeats = utils.to_reference(repeats)
    ref_out = torch.repeat_interleave(ref_repeats)

    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(repeats)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.repeat_interleave_self_tensor
@pytest.mark.parametrize("shape", REPEAT_INTERLEAVE_SHAPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_repeat_interleave_self_tensor(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    repeats = torch.randint(0, 30, (shape[dim],), device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_repeats = utils.to_reference(repeats)

    ref_out = torch.repeat_interleave(ref_inp, ref_repeats, dim)
    with flag_gems.use_gems():
        res_out = torch.repeat_interleave(inp, repeats, dim)

    utils.gems_assert_equal(res_out, ref_out)
