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

_SHAPE_TRIPLES = [
    (2, 19, 7),
    (1, 1, 1),
    (1, 256, 256),
    (4, 1024, 1024),
    (1024, 1024, 64),
    (20, 320, 15),
    (16, 128, 64),
    (16, 7, 57),
]

SUPPORTED_DTYPES = [torch.float16, torch.float32]
if utils.bf16_is_supported:
    SUPPORTED_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    SUPPORTED_DTYPES.append(torch.float64)

# The native cell accepts 2-D input; triples give batch, input size and hidden size.
VALUE_CASES = [
    (shape, value_range, False)
    for shape in tu.selected_cases(_SHAPE_TRIPLES, quick=[(2, 19, 7)])
    for value_range in tu.selected_ranges()
]
VALUE_CASES += tu.selected_cases(
    [
        ((0, 8, 4), ["-1", "1"], False),
        ((4, 0, 8), ["-1", "1"], False),
        ((4, 8, 0), ["-1", "1"], False),
        ((2, 19, 7), ["-1", "1"], True),
    ],
    quick=[],
)

BIAS_CASES = tu.selected_cases([(False, False), (False, True), (True, True)], quick=[])

HX_FORMS = tu.selected_cases([tuple], quick=[])

BACKWARD_TRIPLES = tu.selected_cases([(2, 19, 7), (16, 128, 64)], quick=[])

SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(SUPPORTED_DTYPES),
    quick=[],
)


def _operands(shape, dtype, value_range):
    batch, input_size, hidden = shape
    inp = tu.make_input(dtype, (batch, input_size), value_range)
    hx = [
        tu.make_input(dtype, (batch, hidden), value_range),
        tu.make_input(dtype, (batch, hidden), value_range),
    ]
    w_ih = tu.make_input(dtype, (4 * hidden, input_size), value_range)
    w_hh = tu.make_input(dtype, (4 * hidden, hidden), value_range)
    return inp, hx, w_ih, w_hh


def _special_operand(dtype, scenario, shape):
    payload = tu.make_special_input(dtype, scenario).reshape(-1)
    count = 1
    for dim in shape:
        count *= dim
    repeats = (count + payload.numel() - 1) // payload.numel()
    return payload.repeat(repeats)[:count].reshape(shape)


@pytest.mark.lstm_cell
@pytest.mark.parametrize("shape,value_range,strided", VALUE_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_lstm_cell(shape, value_range, strided, dtype):
    inp, hx, w_ih, w_hh = _operands(shape, dtype, value_range)
    if strided:
        inp, h0, c0, w_ih, w_hh = [
            x.t().contiguous().t() for x in (inp, *hx, w_ih, w_hh)
        ]
        hx = [h0, c0]
    ref_inp = tu.to_reference(inp)
    ref_hx = [tu.to_reference(state) for state in hx]
    ref_w_ih = tu.to_reference(w_ih)
    ref_w_hh = tu.to_reference(w_hh)

    ref_hy, ref_cy = torch.ops.aten.lstm_cell(ref_inp, ref_hx, ref_w_ih, ref_w_hh)
    res_hy, res_cy = flag_gems.lstm_cell(inp, hx, w_ih, w_hh)

    assert res_hy.device == inp.device and res_cy.device == inp.device
    tu.assert_result_close(res_hy, ref_hy)
    tu.assert_result_close(res_cy, ref_cy)


@pytest.mark.lstm_cell
@pytest.mark.parametrize("has_b_ih,has_b_hh", BIAS_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_lstm_cell_bias(has_b_ih, has_b_hh, dtype):
    shape = (20, 320, 15)
    gate_size = 4 * shape[2]
    inp, hx, w_ih, w_hh = _operands(shape, dtype, ["-1", "1"])
    b_ih = tu.make_input(dtype, (gate_size,), ["-1", "1"]) if has_b_ih else None
    b_hh = tu.make_input(dtype, (gate_size,), ["-1", "1"]) if has_b_hh else None

    ref_hy, ref_cy = torch.ops.aten.lstm_cell(
        tu.to_reference(inp),
        [tu.to_reference(state) for state in hx],
        tu.to_reference(w_ih),
        tu.to_reference(w_hh),
        tu.to_reference(b_ih),
        tu.to_reference(b_hh),
    )
    res_hy, res_cy = flag_gems.lstm_cell(inp, hx, w_ih, w_hh, b_ih, b_hh)

    tu.assert_result_close(res_hy, ref_hy)
    tu.assert_result_close(res_cy, ref_cy)


@pytest.mark.lstm_cell
@pytest.mark.parametrize("hx_form", HX_FORMS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_lstm_cell_hx_form(hx_form, dtype):
    inp, hx, w_ih, w_hh = _operands((16, 128, 64), dtype, ["-1", "1"])
    hx = hx_form(hx)

    ref_hx = type(hx)(tu.to_reference(state) for state in hx)
    ref_hy, ref_cy = torch.ops.aten.lstm_cell(
        tu.to_reference(inp), ref_hx, tu.to_reference(w_ih), tu.to_reference(w_hh)
    )
    res_hy, res_cy = flag_gems.lstm_cell(inp, hx, w_ih, w_hh)

    tu.assert_result_close(res_hy, ref_hy)
    tu.assert_result_close(res_cy, ref_cy)


@pytest.mark.lstm_cell
@pytest.mark.parametrize("shape", BACKWARD_TRIPLES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_lstm_cell_backward(shape, dtype):
    batch, input_size, hidden = shape
    gates = 4 * hidden
    inp = tu.make_input(dtype, (batch, input_size), ["-1", "1"]).requires_grad_(True)
    h0 = tu.make_input(dtype, (batch, hidden), ["-1", "1"]).requires_grad_(True)
    c0 = tu.make_input(dtype, (batch, hidden), ["-1", "1"]).requires_grad_(True)
    w_ih = tu.make_input(dtype, (gates, input_size), ["-1", "1"]).requires_grad_(True)
    w_hh = tu.make_input(dtype, (gates, hidden), ["-1", "1"]).requires_grad_(True)
    b_ih = tu.make_input(dtype, (gates,), ["-1", "1"]).requires_grad_(True)
    b_hh = tu.make_input(dtype, (gates,), ["-1", "1"]).requires_grad_(True)

    ref_inp = tu.to_reference(inp)
    ref_h0 = tu.to_reference(h0)
    ref_c0 = tu.to_reference(c0)
    ref_w_ih = tu.to_reference(w_ih)
    ref_w_hh = tu.to_reference(w_hh)
    ref_b_ih = tu.to_reference(b_ih)
    ref_b_hh = tu.to_reference(b_hh)

    ref_hy, ref_cy = torch.ops.aten.lstm_cell(
        ref_inp, [ref_h0, ref_c0], ref_w_ih, ref_w_hh, ref_b_ih, ref_b_hh
    )
    res_hy, res_cy = flag_gems.lstm_cell(inp, [h0, c0], w_ih, w_hh, b_ih, b_hh)

    upstream = [
        tu.make_input(dtype, (batch, hidden), ["-1", "1"]),
        tu.make_input(dtype, (batch, hidden), ["-1", "1"]),
    ]
    ref_upstream = [tu.to_reference(grad) for grad in upstream]

    ref_grads = torch.autograd.grad(
        (ref_hy, ref_cy),
        (ref_inp, ref_h0, ref_c0, ref_w_ih, ref_w_hh, ref_b_ih, ref_b_hh),
        grad_outputs=ref_upstream,
    )
    res_grads = torch.autograd.grad(
        (res_hy, res_cy),
        (inp, h0, c0, w_ih, w_hh, b_ih, b_hh),
        grad_outputs=upstream,
    )

    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.lstm_cell
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_lstm_cell_special_values(dtype, scenario):
    batch, input_size, hidden = 1, 5, 3
    inp = _special_operand(dtype, scenario, (batch, input_size))
    h0 = _special_operand(dtype, scenario, (batch, hidden))
    c0 = _special_operand(dtype, scenario, (batch, hidden))
    w_ih = _special_operand(dtype, scenario, (4 * hidden, input_size))
    w_hh = _special_operand(dtype, scenario, (4 * hidden, hidden))

    ref_hy, ref_cy = torch.ops.aten.lstm_cell(
        tu.to_reference(inp),
        [tu.to_reference(h0), tu.to_reference(c0)],
        tu.to_reference(w_ih),
        tu.to_reference(w_hh),
    )
    res_hy, res_cy = flag_gems.lstm_cell(inp, [h0, c0], w_ih, w_hh)

    tu.assert_result_close(res_hy, ref_hy)
    tu.assert_result_close(res_cy, ref_cy)


@pytest.mark.lstm_cell
def test_lstm_cell_negative_input_rank():
    inp, hx, w_ih, w_hh = _operands((3, 8, 4), torch.float32, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        flag_gems.lstm_cell(inp.unsqueeze(0), hx, w_ih, w_hh)


@pytest.mark.lstm_cell
def test_lstm_cell_negative_hx_not_sequence():
    inp, hx, w_ih, w_hh = _operands((3, 8, 4), torch.float32, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        flag_gems.lstm_cell(inp, hx[0], w_ih, w_hh)


@pytest.mark.lstm_cell
def test_lstm_cell_negative_hx_length():
    inp, hx, w_ih, w_hh = _operands((3, 8, 4), torch.float32, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        flag_gems.lstm_cell(inp, hx[:1], w_ih, w_hh)


@pytest.mark.lstm_cell
@pytest.mark.parametrize(
    "shape", tu.selected_cases([(1024, 1024, 64)], quick=[(2, 19, 7)])
)
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in SUPPORTED_DTYPES if dtype in (torch.float16, torch.bfloat16)],
)
def test_lstm_cell_negative_input_bias_only(shape, dtype):
    # Native CUDA lstm_cell requires b_hh when b_ih is present.
    inp, hx, w_ih, w_hh = _operands(shape, dtype, ["-1", "1"])
    b_ih = tu.make_input(dtype, (w_ih.shape[0],), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.lstm_cell(inp, hx, w_ih, w_hh, b_ih)


@pytest.mark.lstm_cell
def test_lstm_cell_negative_bias_rank():
    inp, hx, w_ih, w_hh = _operands((3, 8, 4), torch.float32, ["-1", "1"])
    bad_b_ih = tu.make_input(torch.float32, (2, w_ih.shape[0]), ["-1", "1"])
    good_b_hh = tu.make_input(torch.float32, (w_hh.shape[0],), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        flag_gems.lstm_cell(inp, hx, w_ih, w_hh, bad_b_ih, good_b_hh)


@pytest.mark.lstm_cell
@pytest.mark.parametrize(
    "shape", tu.selected_cases([(1024, 1024, 64)], quick=[(2, 19, 7)])
)
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in SUPPORTED_DTYPES if dtype in (torch.float16, torch.bfloat16)],
)
@pytest.mark.parametrize("bias", ["b_ih", "b_hh"])
@pytest.mark.parametrize("length_delta", [-1, 1])
def test_lstm_cell_negative_bias_length(shape, dtype, bias, length_delta):
    inp, hx, w_ih, w_hh = _operands(shape, dtype, ["-1", "1"])
    gates = w_ih.shape[0]
    b_ih = tu.make_input(
        dtype, (gates + (length_delta if bias == "b_ih" else 0),), ["-1", "1"]
    )
    b_hh = tu.make_input(
        dtype, (gates + (length_delta if bias == "b_hh" else 0),), ["-1", "1"]
    )
    with pytest.raises(RuntimeError):
        flag_gems.lstm_cell(inp, hx, w_ih, w_hh, b_ih, b_hh)


@pytest.mark.lstm_cell
def test_lstm_cell_negative_batch_mismatch():
    inp, hx, w_ih, w_hh = _operands((3, 8, 4), torch.float32, ["-1", "1"])
    bad_h0 = tu.make_input(
        torch.float32, (inp.shape[0] + 1, hx[1].shape[1]), ["-1", "1"]
    )
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        flag_gems.lstm_cell(inp, [bad_h0, hx[1]], w_ih, w_hh)


@pytest.mark.lstm_cell
def test_lstm_cell_negative_weight_shape():
    inp, hx, w_ih, w_hh = _operands((3, 8, 4), torch.float32, ["-1", "1"])
    bad_w_hh = tu.make_input(
        torch.float32, (w_hh.shape[0], w_hh.shape[1] + 3), ["-1", "1"]
    )
    with pytest.raises((RuntimeError, TypeError, ValueError, IndexError)):
        flag_gems.lstm_cell(inp, hx, w_ih, bad_w_hh)


@pytest.mark.lstm_cell
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ],
)
def test_lstm_cell_negative_unsupported_dtype(dtype):
    inp, hx, w_ih, w_hh = _operands((3, 8, 4), dtype, ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError, ValueError, NotImplementedError)):
        flag_gems.lstm_cell(inp, hx, w_ih, w_hh)
