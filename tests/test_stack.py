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

if flag_gems.vendor_name == "cambricon":
    CAMBRICON_STACK_SHAPES = [
        [
            (8, 8, 128),
            (8, 8, 128),
            (8, 8, 128),
        ],
        [
            (32, 64, 128, 8),
            (32, 64, 128, 8),
            (32, 64, 128, 8),
            (32, 64, 128, 8),
        ],
    ]

    STACK_SHAPES_TEST = utils.STACK_SHAPES + CAMBRICON_STACK_SHAPES
else:
    STACK_SHAPES_TEST = utils.STACK_SHAPES


@pytest.mark.stack
@pytest.mark.parametrize("shape", STACK_SHAPES_TEST)
@pytest.mark.parametrize("dim", utils.STACK_DIM_LIST)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test_stack(shape, dim, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]

    ref_inp = [utils.to_reference(_) for _ in inp]
    ref_out = torch.stack(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = torch.stack(inp, dim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_stack
@pytest.mark.parametrize("shape", utils.STACK_SHAPES)
@pytest.mark.parametrize("dim", utils.STACK_DIM_LIST)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
def test__stack(shape, dim, dtype):
    if dtype in utils.FLOAT_DTYPES:
        inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]
    else:
        inp = [
            torch.randint(low=0, high=0x7FFF, size=s, dtype=dtype, device="cpu").to(
                flag_gems.device
            )
            for s in shape
        ]

    ref_inp = [utils.to_reference(_) for _ in inp]
    ref_out = torch._stack(ref_inp, dim)

    res_out = flag_gems._stack(inp, dim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_stack
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__stack_dim_out_of_range(dtype):
    inp = [torch.randn(3, 4, dtype=dtype, device=flag_gems.device) for _ in range(2)]
    with pytest.raises(IndexError):
        flag_gems._stack(inp, 4)


@pytest.mark.underscore_stack
@pytest.mark.parametrize("shape", utils.STACK_SHAPES)
@pytest.mark.parametrize("dim", utils.STACK_DIM_LIST)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test__stack_complex(shape, dim, dtype):
    inp = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shape]

    ref_inp = [utils.to_reference(_) for _ in inp]
    ref_out = torch._stack(ref_inp, dim)

    res_out = flag_gems._stack(inp, dim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_stack
def test__stack_dim_out_of_range_single_input():
    # With a single tensor the per-entry loop never runs; an out-of-range dim
    # must still be rejected (matches ATen).
    inp = [torch.randn(3, 4, dtype=torch.float32, device=flag_gems.device)]
    with pytest.raises(IndexError):
        flag_gems._stack(inp, 4)


@pytest.mark.underscore_stack
@pytest.mark.parametrize("shape", utils.STACK_SHAPES)
@pytest.mark.parametrize("dim", utils.STACK_DIM_LIST)
def test__stack_zero_sized(shape, dim):
    # ``shape`` is a list of per-tensor shapes; make the trailing dimension
    # zero so the output has zero volume while strides stay non-zero.
    zero_shape = list(shape[0])
    zero_shape[-1] = 0
    inp = [
        torch.empty(zero_shape, dtype=torch.float32, device=flag_gems.device)
        for _ in range(3)
    ]

    ref_inp = [utils.to_reference(_) for _ in inp]
    ref_out = torch._stack(ref_inp, dim)

    res_out = flag_gems._stack(inp, dim)

    assert res_out.shape == ref_out.shape
    assert res_out.numel() == 0
