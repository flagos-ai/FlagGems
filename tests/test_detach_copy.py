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
from . import test_utils as tu

# Copy values into independent contiguous storage; backward is unsupported.
_DETACH_COPY_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bool,
        torch.complex64,
    ]
)

_DETACH_COPY_NONCONTIG_SHAPES = [(8, 16, 32), (4, 8, 16, 32)]
_DETACH_COPY_EMPTY_SHAPES = [(0,), (4, 0), (2, 0, 3)]
_DETACH_COPY_NO_BACKWARD_SHAPES = [(16, 64), (7, 13, 29)]
_DETACH_COPY_STORAGE_SHAPES = [(16, 32), (64, 128)]


def _assert_copy_semantics(res_out, ref_out, inp, ref_inp):
    assert res_out.device == inp.device
    assert res_out.is_contiguous()
    assert res_out.stride() == ref_out.stride()
    # Zero-element tensors carry a null data pointer on every tensor, so the
    # no-alias check is only meaningful for non-empty inputs.
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(res_out, ref_out)
    # The input must be untouched by the copy.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    # to_reference creates an independent snapshot for the input mutation check.
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_detach_copy_special_values(dtype):
    inp = torch.tensor(
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            0.0,
            -0.0,
            1.5,
            -2.5,
            1e30,
            -1e30,
        ],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    tu.assert_result_equal(res_out, ref_out)
    # -0.0 must copy with its sign bit intact (equal_nan-tolerant compares treat
    # -0.0 == 0.0, so pin the sign explicitly).
    assert torch.equal(torch.signbit(res_out), torch.signbit(ref_out))


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_out(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    # Garbage-prefilled out buffers: the .out overload must overwrite them.
    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.detach_copy(inp, out=res_out)

    # The .out overload must write into and return the caller's buffer.
    assert res_ret is res_out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp)


@pytest.mark.detach_copy_out
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_out_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.full(shape, 7, dtype=ref_inp.dtype, device=ref_inp.device)
    res_out = torch.full(shape, 7, dtype=dtype, device=flag_gems.device)

    ref_ret = torch.ops.aten.detach_copy.out(ref_inp, out=ref_out)
    res_ret = flag_gems.detach_copy(inp, out=res_out)

    assert res_ret is res_out
    _assert_copy_semantics(res_ret, ref_ret, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_non_contiguous(shape, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base.transpose(-1, -2)
    ref_inp = ref_base.transpose(-1, -2)
    assert not inp.is_contiguous()

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", _DETACH_COPY_DTYPES)
def test_detach_copy_empty(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    _assert_copy_semantics(res_out, ref_out, inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_STORAGE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_detach_copy_independent_storage(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    res_out = flag_gems.detach_copy(inp)

    tu.assert_result_equal(res_out, ref_out)
    res_out.fill_(3.25)
    if inp.numel() > 0:
        assert res_out.data_ptr() != inp.data_ptr()
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.detach_copy
@pytest.mark.parametrize("shape", _DETACH_COPY_NO_BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES, quick=utils.FLOAT_DTYPES)
)
def test_detach_copy_no_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    grad = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.detach_copy(ref_inp)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    res_out = flag_gems.detach_copy(inp)
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(inp, ref_inp)

    with pytest.raises(RuntimeError):
        torch.autograd.grad(res_out, inp, grad_outputs=grad)


@pytest.mark.detach_copy
def test_detach_copy_rejects_non_tensor():
    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.detach_copy(3.14)


@pytest.mark.detach_copy_out
def test_detach_copy_out_rejects_wrong_dtype():
    inp = tu.make_input(torch.float32, (8,), ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)
    res_out_bad = torch.empty(8, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten.detach_copy.out(ref_inp, out=ref_out_bad)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.detach_copy(inp, out=res_out_bad)


@pytest.mark.detach_copy
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_DETACH_COPY_DTYPES))
)
def test_detach_copy_special_scenarios(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    reference = tu.to_reference(inp)
    expected = torch.ops.aten.detach_copy(reference)
    actual = flag_gems.detach_copy(inp)
    tu.assert_result_equal(actual, expected)
