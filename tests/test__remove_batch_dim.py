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
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
setattr(
    pytest.mark,
    "_remove_batch_dim",
    MarkDecorator(Mark("_remove_batch_dim", (), {}, _ispytest=True), _ispytest=True),
)

# For plain tensors, insert batch_size into the target shape and expand.
_REMOVE_BATCH_DIM_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.bool, torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
)

# (shape, out_dim, batch_size), including front/middle/end insertion and size-1 broadcast.
_REMOVE_BATCH_DIM_CASES = [
    ((), 0, 7),  # rank-0: batch becomes the only dim
    ((16,), 0, 7),  # rank-1, out_dim at the front
    ((16,), 1, 16),  # rank-1, out_dim at the end (batch == s0)
    ((1,), 0, 5),  # rank-1, size-1 dim at the front
    ((1,), 1, 9),  # rank-1, size-1 dim broadcast at the end
    ((256,), 0, 11),  # rank-1 regular 1-D shape, front
    ((256,), 1, 256),  # rank-1, out_dim at the end (batch == s0)
    ((64, 32), 0, 13),  # rank-2, out_dim at the front
    ((64, 32), 1, 64),  # rank-2, batch matches dim0
    ((1, 32), 1, 7),  # rank-2, size-1 broadcast at dim0
    ((1024, 1024), 0, 1),  # rank-2 regular 2-D shape, batch size 1
    ((2, 19, 7), 0, 5),  # rank-3, out_dim at the front
    ((2, 19, 7), 1, 2),  # rank-3, batch matches dim0
    ((1, 19, 7), 1, 5),  # rank-3, size-1 broadcast at dim0
    ((1, 19, 7), 2, 19),  # rank-3, middle out_dim, size-1 dim0
    ((4, 4, 16), 2, 4),  # rank-3, middle out_dim, adjacent dims equal
    ((20, 320, 15), 0, 1),  # rank-3 spec shape, out_dim at the front
    ((20, 320, 15), 1, 20),  # rank-3 spec shape, batch matches dim0
    ((4, 8, 16, 32), 0, 9),  # rank-4, out_dim at the front
    ((4, 8, 16, 32), 1, 4),  # rank-4, batch matches dim0
    ((8, 8, 8, 32), 2, 8),  # rank-4, middle out_dim
    ((1, 8, 16, 32), 2, 8),  # rank-4, middle out_dim, size-1 dim0
    ((16, 128, 64, 60), 0, 1),  # rank-4 spec shape, batch size 1
    ((16, 7, 57, 32, 29), 0, 1),  # rank-5, out_dim at the front
    ((1, 7, 57, 32, 29), 1, 11),  # rank-5, size-1 dim0 broadcast
]

_BACKWARD_CASES = [
    ((2, 19, 7), 1, 2),
    ((4, 8, 16, 32), 0, 9),
    ((1, 19, 7), 2, 19),
]

_NON_CONTIGUOUS_CASES = [
    ((8, 16, 32), 1, 8),
    ((8, 8, 16, 32), 2, 8),
]

_DTYPE_RANGE_PAIRS = [
    (dtype, value_range)
    for dtype in _REMOVE_BATCH_DIM_DTYPES
    for value_range in tu.selected_ranges()
]


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("shape, out_dim, batch_size", _REMOVE_BATCH_DIM_CASES)
@pytest.mark.parametrize("level", [0, 1, 3])
@pytest.mark.parametrize("dtype", _REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim(shape, out_dim, batch_size, level, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = flag_gems._remove_batch_dim(inp, level, batch_size, out_dim)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype, value_range", _DTYPE_RANGE_PAIRS)
def test__remove_batch_dim_value_ranges(shape, dtype, value_range):
    out_dim, batch_size = 0, 1
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, 0, batch_size, out_dim)
    res_out = flag_gems._remove_batch_dim(inp, 0, batch_size, out_dim)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("dtype, value_range", tu.selected_cases(_DTYPE_RANGE_PAIRS))
def test__remove_batch_dim_value_ranges_broadcast(dtype, value_range):
    shape, out_dim, batch_size = (2, 19, 7), 1, 2
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, 0, batch_size, out_dim)
    res_out = flag_gems._remove_batch_dim(inp, 0, batch_size, out_dim)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("shape, out_dim, batch_size", _NON_CONTIGUOUS_CASES)
@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("dtype", _REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim_non_contiguous(shape, out_dim, batch_size, level, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = flag_gems._remove_batch_dim(inp, level, batch_size, out_dim)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.FLOAT_DTYPES))
def test__remove_batch_dim_nan_inf(dtype):
    vals = [
        float("inf"),
        float("-inf"),
        float("nan"),
        0.0,
        -0.0,
        1.5,
        -2.5,
        1e30,
        -1e30,
    ]
    inp = torch.tensor(vals, dtype=dtype, device=flag_gems.device).reshape(3, 3)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, 0, 4, 0)
    res_out = flag_gems._remove_batch_dim(inp, 0, 4, 0)

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("shape, out_dim, batch_size", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test__remove_batch_dim_backward(shape, out_dim, batch_size, dtype):
    level = 0
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    out_shape = list(shape)
    out_shape.insert(out_dim, batch_size)
    out_shape = tuple(out_shape)
    grad_out = tu.make_input(dtype, out_shape, ["-1", "1"])
    ref_grad_out = tu.to_reference(grad_out)

    inp.requires_grad_(True)
    ref_inp.requires_grad_(True)

    ref_out = torch.ops.aten._remove_batch_dim(ref_inp, level, batch_size, out_dim)
    res_out = flag_gems._remove_batch_dim(inp, level, batch_size, out_dim)
    tu.assert_result_equal(res_out, ref_out)

    res_grad = torch.autograd.grad(res_out, inp, grad_out)[0]
    ref_grad = torch.autograd.grad(ref_out, ref_inp, ref_grad_out)[0]

    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark._remove_batch_dim
def test__remove_batch_dim_rejects_non_broadcastable_batch_size():
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._remove_batch_dim(inp, 0, 3, 1)
    with pytest.raises(RuntimeError):
        flag_gems._remove_batch_dim(inp, 0, 3, 1)


@pytest.mark._remove_batch_dim
def test__remove_batch_dim_rejects_negative_batch_size():
    inp = tu.make_input(torch.float32, (2, 19, 7), ["-1", "1"])
    with pytest.raises(RuntimeError):
        torch.ops.aten._remove_batch_dim(inp, 0, -1, 0)
    with pytest.raises(RuntimeError):
        flag_gems._remove_batch_dim(inp, 0, -1, 0)


@pytest.mark._remove_batch_dim
def test__remove_batch_dim_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten._remove_batch_dim(3.14, 0, 1, 0)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems._remove_batch_dim(3.14, 0, 1, 0)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize(
    "dtype, scenario",
    tu.selected_cases(tu.special_value_cases(_REMOVE_BATCH_DIM_DTYPES)),
)
def test__remove_batch_dim_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten._remove_batch_dim(reference, 0, 4, 0)
    actual = flag_gems._remove_batch_dim(inp, 0, 4, 0)
    tu.assert_result_equal(actual, expected)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("batch_dim, out_dim", [(0, 0), (0, 2), (1, 0), (1, 2), (2, 1)])
@pytest.mark.parametrize("level", [0, 3])
@pytest.mark.parametrize("dtype", _REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim_batched(batch_dim, out_dim, level, dtype):
    inp = tu.make_input(dtype, (3, 5, 14), ["-1", "1"])[..., 1::2]
    ref_inp = tu.to_reference(inp)
    batched = torch.ops.aten._add_batch_dim(inp, batch_dim, level)
    ref_batched = torch.ops.aten._add_batch_dim(ref_inp, batch_dim, level)
    batch_size = inp.size(batch_dim)

    ref_out = torch.ops.aten._remove_batch_dim(ref_batched, level, batch_size, out_dim)
    res_out = flag_gems._remove_batch_dim(batched, level, batch_size, out_dim)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert torch._C._is_alias_of(res_out, inp)


@pytest.mark._remove_batch_dim
@pytest.mark.parametrize("dtype", _REMOVE_BATCH_DIM_DTYPES)
def test__remove_batch_dim_other_level(dtype):
    inp = tu.make_input(dtype, (3, 5), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    batched = torch.ops.aten._add_batch_dim(inp, 0, 0)
    ref_batched = torch.ops.aten._add_batch_dim(ref_inp, 0, 0)
    ref_out = torch.ops.aten._remove_batch_dim(ref_batched, 1, 2, 0)
    res_out = flag_gems._remove_batch_dim(batched, 1, 2, 0)

    assert torch._C._functorch.is_legacy_batchedtensor(res_out)
    ref_physical = torch.ops.aten._remove_batch_dim(ref_out, 0, 3, 0)
    res_physical = torch.ops.aten._remove_batch_dim(res_out, 0, 3, 0)
    tu.assert_result_equal(res_physical, ref_physical)
    assert torch._C._is_alias_of(res_physical, inp)
