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

GRU_CELL_SHAPES = tu.selected_cases(
    [
        (2, 3, 4),
        (1, 256, 128),
        (64, 64, 64),
        (16, 128, 32),
        (4, 320, 160),
        (7, 57, 21),
        (128, 512, 256),
        (1024, 1024, 64),
    ],
    quick=[(2, 19, 7)],
)

GRU_CELL_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    GRU_CELL_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    GRU_CELL_DTYPES.append(torch.float64)
UNSUPPORTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int32,
    torch.int64,
    torch.bool,
]


@pytest.fixture(autouse=True)
def ieee_precision(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)


VALUE_CASES = [
    (shape, value_range, False)
    for shape in GRU_CELL_SHAPES
    for value_range in tu.selected_ranges()
]
VALUE_CASES += tu.selected_cases(
    [
        ((4, 5, 6), ["-1", "1"], True),
        ((32, 128, 64), ["-1", "1"], True),
        ((0, 8, 4), ["-1", "1"], False),
        ((4, 0, 8), ["-1", "1"], False),
        ((4, 8, 0), ["-1", "1"], False),
    ],
    quick=[],
)


def _make_operands(dtype, shape, value_range, with_bias=False):
    batch, input_size, hidden_size = shape
    inp = tu.make_input(dtype, (batch, input_size), value_range)
    hx = tu.make_input(dtype, (batch, hidden_size), value_range)
    w_ih = tu.make_input(dtype, (3 * hidden_size, input_size), value_range)
    w_hh = tu.make_input(dtype, (3 * hidden_size, hidden_size), value_range)
    if not with_bias:
        return inp, hx, w_ih, w_hh
    b_ih = tu.make_input(dtype, (3 * hidden_size,), value_range)
    b_hh = tu.make_input(dtype, (3 * hidden_size,), value_range)
    return inp, hx, w_ih, w_hh, b_ih, b_hh


@pytest.mark.gru_cell
@pytest.mark.parametrize("shape,value_range,with_bias", VALUE_CASES)
@pytest.mark.parametrize("dtype", GRU_CELL_DTYPES)
def test_gru_cell(shape, value_range, with_bias, dtype):
    operands = _make_operands(dtype, shape, value_range, with_bias=with_bias)
    reference = tuple(tu.to_reference(tensor) for tensor in operands)

    ref_out = torch.ops.aten.gru_cell(*reference)
    res_out = flag_gems.gru_cell(*operands)

    tu.assert_result_close(res_out, ref_out)


_STRIDED_LAYOUTS = tu.selected_cases(
    [((8, 12), slice(3, 7), slice(2, 7)), ((8, 12), slice(1, 9, 2), slice(0, 12, 2))],
    quick=[],
)


@pytest.mark.gru_cell
@pytest.mark.parametrize("base_shape,rows,cols", _STRIDED_LAYOUTS)
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in GRU_CELL_DTYPES if dtype in (torch.float32, torch.bfloat16)],
)
def test_gru_cell_strided_input(base_shape, rows, cols, dtype):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    inp = base[rows, cols]
    batch, input_size = inp.shape
    hidden_size = 4
    operands = (
        inp,
        tu.make_input(dtype, (batch, hidden_size), ["-1", "1"]),
        tu.make_input(dtype, (3 * hidden_size, input_size), ["-1", "1"]),
        tu.make_input(dtype, (3 * hidden_size, hidden_size), ["-1", "1"]),
    )
    reference = tuple(tu.to_reference(tensor) for tensor in operands)

    ref_out = torch.ops.aten.gru_cell(*reference)
    res_out = flag_gems.gru_cell(*operands)

    tu.assert_result_close(res_out, ref_out)


_BACKWARD_SHAPES = tu.selected_cases([(4, 5, 6)], quick=[])


@pytest.mark.gru_cell
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dtype", GRU_CELL_DTYPES)
def test_gru_cell_backward(shape, with_bias, dtype):
    operands = _make_operands(dtype, shape, ["-1", "1"], with_bias=with_bias)
    leaves = tuple(tensor.detach().requires_grad_(True) for tensor in operands)
    reference = tuple(
        tu.to_reference(tensor).detach().requires_grad_(True) for tensor in operands
    )

    ref_out = torch.ops.aten.gru_cell(*reference)
    res_out = flag_gems.gru_cell(*leaves)
    tu.assert_result_close(res_out, ref_out)

    upstream = tu.make_input(dtype, ref_out.shape, ["-1", "1"])
    ref_grads = torch.autograd.grad(
        ref_out, reference, grad_outputs=tu.to_reference(upstream)
    )
    res_grads = torch.autograd.grad(res_out, leaves, grad_outputs=upstream)
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(GRU_CELL_DTYPES), quick=[])


@pytest.mark.gru_cell
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_gru_cell_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    operands = (
        payload.repeat(4).view(4, 5),
        payload.repeat(4).view(4, 5),
        payload.repeat(15).view(15, 5),
        payload.repeat(15).view(15, 5),
    )
    reference = tuple(tu.to_reference(tensor) for tensor in operands)

    ref_out = torch.ops.aten.gru_cell(*reference)
    res_out = flag_gems.gru_cell(*operands)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.gru_cell
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test_gru_cell_unsupported_dtype(dtype):
    operands = _make_operands(dtype, (2, 3, 4), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.gru_cell(*operands)


_MALFORMED_SHAPES = [
    ((2, 3, 4), (2, 4), (12, 3), (12, 4)),
    ((2, 3), (4,), (12, 3), (12, 4)),
    ((2, 3), (), (12, 3), (12, 4)),
    ((2, 3), (2, 4), (12, 3), (4,)),
    ((2, 3), (2, 4), (8, 3), (12, 4)),
    ((2, 3), (5, 4), (12, 3), (12, 4)),
]


@pytest.mark.gru_cell
@pytest.mark.parametrize("shapes", _MALFORMED_SHAPES)
def test_gru_cell_malformed_operands(shapes):
    inp_shape, hx_shape, w_ih_shape, w_hh_shape = shapes
    operands = (
        tu.make_input(torch.float32, inp_shape, ["-1", "1"]),
        tu.make_input(torch.float32, hx_shape, ["-1", "1"]),
        tu.make_input(torch.float32, w_ih_shape, ["-1", "1"]),
        tu.make_input(torch.float32, w_hh_shape, ["-1", "1"]),
    )
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.gru_cell(*operands)


@pytest.mark.gru_cell
@pytest.mark.parametrize(
    "shape", tu.selected_cases([(1024, 1024, 64)], quick=[(2, 19, 7)])
)
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in GRU_CELL_DTYPES if dtype in (torch.float16, torch.bfloat16)],
)
@pytest.mark.parametrize("bias", ["b_ih", "b_hh"])
@pytest.mark.parametrize("length_delta", [-1, 1])
def test_gru_cell_bias_length_mismatch(shape, dtype, bias, length_delta):
    inp, hx, w_ih, w_hh = _make_operands(dtype, shape, ["-1", "1"])
    gates = w_ih.shape[0]
    b_ih = tu.make_input(
        dtype, (gates + (length_delta if bias == "b_ih" else 0),), ["-1", "1"]
    )
    b_hh = tu.make_input(
        dtype, (gates + (length_delta if bias == "b_hh" else 0),), ["-1", "1"]
    )
    with pytest.raises(RuntimeError):
        flag_gems.gru_cell(inp, hx, w_ih, w_hh, b_ih, b_hh)


@pytest.mark.gru_cell
def test_gru_cell_partial_bias():
    inp, hx, w_ih, w_hh = _make_operands(torch.float32, (2, 3, 4), ["-1", "1"])
    b_ih = tu.make_input(torch.float32, (12,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.gru_cell(inp, hx, w_ih, w_hh, b_ih)


@pytest.mark.gru_cell
@pytest.mark.parametrize("dtype", tu.selected_cases(GRU_CELL_DTYPES, quick=[]))
def test_gru_cell_hidden_bias_only(dtype):
    inp, hx, w_ih, w_hh = _make_operands(dtype, (20, 320, 15), ["-1", "1"])
    b_hh = tu.make_input(dtype, (45,), ["-1", "1"])
    ref_out = torch.ops.aten.gru_cell(
        tu.to_reference(inp),
        tu.to_reference(hx),
        tu.to_reference(w_ih),
        tu.to_reference(w_hh),
        b_hh=tu.to_reference(b_hh),
    )
    res_out = flag_gems.gru_cell(inp, hx, w_ih, w_hh, b_hh=b_hh)
    tu.assert_result_close(res_out, ref_out)
