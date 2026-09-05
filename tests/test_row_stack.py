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

# Core shapes exercised by ROW_STACK_SHAPES (mirrors worktree CI branch).
ROW_STACK_SHAPES = [
    [(3,), (3,)],
    [(3, 33), (7, 33)],
    [(13, 3, 333), (17, 3, 333), (7, 3, 333)],
    [
        (13, 3, 64, 5, 2),
        (16, 3, 64, 5, 2),
        (7, 3, 64, 5, 2),
        (4, 3, 64, 5, 2),
        (1, 3, 64, 5, 2),
    ],
]


def _make_inputs(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        return [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        return [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]


@pytest.mark.row_stack
@pytest.mark.parametrize("shape", ROW_STACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_row_stack(shape, dtype):
    inp = _make_inputs(shape, dtype)
    ref_inp = [utils.to_reference(e) for e in inp]
    # Reference via the public aten op on the reference (CPU) inputs.
    ref_out = torch.row_stack(ref_inp)
    # GEMS direct call: the kernel vertically stacks on the accelerator.
    res_out = flag_gems.row_stack(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.row_stack_out
@pytest.mark.parametrize("shape", ROW_STACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_row_stack_out(shape, dtype):
    inp = _make_inputs(shape, dtype)
    ref_inp = [utils.to_reference(e) for e in inp]

    # Compute the reference output to derive the expected out shape.
    ref_out = torch.row_stack(ref_inp)

    # Build an out tensor with the expected shape on the gems device and
    # invoke the out variant directly. The kernel writes into ``res_out``.
    res_out = torch.empty(ref_out.shape, dtype=dtype, device=flag_gems.device)
    flag_gems.row_stack_out(inp, out=res_out)

    utils.gems_assert_equal(res_out, ref_out)
