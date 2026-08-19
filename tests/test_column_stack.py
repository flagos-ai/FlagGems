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

# Each entry is a list of tensor shapes passed together to column_stack.
# Shapes are chosen to cover the distinct code paths of the op: 1-D inputs
# (reshaped into columns), 2-D inputs stacked along dim=1, mixed 1-D/2-D
# inputs, and higher-rank inputs whose trailing dims must be preserved.
COLUMN_STACK_SHAPES = [
    # 1-D tensors get reshaped into (numel, 1) columns.
    [(8,), (8,)],
    [(1024,), (1024,), (1024,)],
    # 2-D tensors are stacked along dim=1.
    [(16, 256), (16, 128)],
    [(64, 32), (64, 16), (64, 8)],
    # Mix of 1-D and 2-D tensors.
    [(5,), (5, 2), (5, 2)],
    # Higher rank tensors keep their trailing dims.
    [(20, 320, 15), (20, 160, 15), (20, 80, 15)],
]
# Shapes that must raise: mismatched non-concatenated dims (rows differ) and
# mismatched 1-D lengths, exercising the wrapper's shape-validation path.
COLUMN_STACK_EXCEPTION_SHAPES = [
    [(16, 256), (8, 128)],
    [(5,), (6,)],
]


def _make_inputs(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        return [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    return [
        torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        for s in shape
    ]


@pytest.mark.column_stack
@pytest.mark.parametrize("shape", COLUMN_STACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_column_stack(shape, dtype):
    inp = _make_inputs(shape, dtype)
    ref_inp = [utils.to_reference(_) for _ in inp]

    ref_out = torch.column_stack(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.column_stack(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.column_stack_out
@pytest.mark.parametrize("shape", COLUMN_STACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_column_stack_out(shape, dtype):
    inp = _make_inputs(shape, dtype)
    ref_inp = [utils.to_reference(_) for _ in inp]

    ref_out = torch.column_stack(ref_inp)
    res_out = torch.empty_like(ref_out, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.column_stack(inp, out=res_out)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.column_stack
@pytest.mark.parametrize("shape", COLUMN_STACK_EXCEPTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exception_column_stack(shape, dtype):
    inp = _make_inputs(shape, dtype)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            _ = torch.column_stack(inp)
