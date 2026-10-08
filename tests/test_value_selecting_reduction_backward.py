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
    # quick mode only exercises float32 to keep CI fast
    FLOAT_DTYPES = [torch.float32]
    # (grad_shape, original input sizes); single small case in quick mode
    VALUE_SELECTING_REDUCTION_BACKWARD_SHAPES = [((3,), (3, 4))]
    DIM_LIST = [1]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    # (grad_shape, original input sizes); keep grad shapes aligned with the
    # reduced sizes so that dim can range over all input dimensions
    VALUE_SELECTING_REDUCTION_BACKWARD_SHAPES = [
        ((3,), (3, 4)),
        ((4,), (4, 8)),
        ((2, 3), (2, 3, 5)),
        ((8, 16), (8, 16, 32)),
    ]
    DIM_LIST = [0, 1, 2]


@pytest.mark.value_selecting_reduction_backward
@pytest.mark.parametrize("grad_shape, sizes", VALUE_SELECTING_REDUCTION_BACKWARD_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_value_selecting_reduction_backward(grad_shape, sizes, dim, keepdim, dtype):
    ndim = len(sizes)
    dim = dim % ndim

    # grad shape follows from the reduced shape of the forward pass
    if keepdim:
        grad_actual_shape = list(sizes)
        grad_actual_shape[dim] = 1
    else:
        grad_actual_shape = list(sizes)
        grad_actual_shape.pop(dim)

    grad = torch.randn(grad_actual_shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad)

    # indices must be valid positions along the reduced dimension
    max_index = sizes[dim]
    indices = torch.randint(0, max_index, grad_actual_shape, device=flag_gems.device)
    ref_indices = utils.to_reference(indices)

    ref_out = torch.ops.aten.value_selecting_reduction_backward(
        ref_grad, dim, ref_indices, list(sizes), keepdim
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.value_selecting_reduction_backward(
            grad, dim, indices, list(sizes), keepdim
        )

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.value_selecting_reduction_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_value_selecting_reduction_backward_max_dim(dtype):
    # realistic max(dim=1) backward: (4, 8) input reduced along dim 1
    inp = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    values, indices = inp.max(dim=1)
    ref_values, ref_indices = ref_inp.max(dim=1)

    grad_output = torch.randn_like(values)
    ref_grad_output = utils.to_reference(grad_output)

    ref_out = torch.ops.aten.value_selecting_reduction_backward(
        ref_grad_output, 1, ref_indices, list(ref_inp.shape), False
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.value_selecting_reduction_backward(
            grad_output, 1, indices, list(inp.shape), False
        )

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.value_selecting_reduction_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_value_selecting_reduction_backward_min_dim(dtype):
    # realistic min(dim=1) backward: (4, 8) input reduced along dim 1
    inp = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    values, indices = inp.min(dim=1)
    ref_values, ref_indices = ref_inp.min(dim=1)

    grad_output = torch.randn_like(values)
    ref_grad_output = utils.to_reference(grad_output)

    ref_out = torch.ops.aten.value_selecting_reduction_backward(
        ref_grad_output, 1, ref_indices, list(ref_inp.shape), False
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.value_selecting_reduction_backward(
            grad_output, 1, indices, list(inp.shape), False
        )

    utils.gems_assert_equal(res_out, ref_out)
