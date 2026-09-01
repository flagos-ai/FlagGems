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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils

# ``_reshape_alias`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._reshape_alias`` and ``-m
# _reshape_alias`` both work.
setattr(
    pytest.mark,
    "_reshape_alias",
    MarkDecorator(Mark("_reshape_alias", (), {}, _ispytest=True), _ispytest=True),
)

# (input_shape, size, stride) triples that describe a valid contiguous reshape
# sharing the same storage as the input.
RESHAPE_ALIAS_CASES = [
    ((3, 4), [4, 3], [3, 1]),
    ((3, 4), [12], [1]),
    ((12,), [3, 4], [4, 1]),
    ((2, 3, 4), [6, 4], [4, 1]),
    ((2, 3, 4), [24], [1]),
    ((4, 4), [2, 2, 4], [8, 4, 1]),
]


@pytest.mark._reshape_alias
@pytest.mark.parametrize("input_shape, size, stride", RESHAPE_ALIAS_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy__reshape_alias(input_shape, size, stride, dtype):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._reshape_alias(ref_inp, size, stride)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._reshape_alias(inp, size, stride)

    assert list(res_out.shape) == list(size)
    assert list(res_out.stride()) == list(stride)
    utils.gems_assert_equal(res_out, ref_out)
