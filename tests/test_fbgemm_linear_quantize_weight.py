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

from . import test_utils as tu

# aten::fbgemm_linear_quantize_weight(input) -> (Tensor, Tensor, float, int)
#   q: int8, same shape as input; bias: int32 with shape input.shape[:-1];
#   scale: Python float; zero_point: Python int -- one pair per tensor, with the
#   integer bias consumed by the packed GEMM kernels.
# Native contract probed on this checkout: rank >= 2 (rank 1/0 raise IndexError
# inside the kernel), float32 only, and no autograd formula for the integer
# outputs. The kernel is CPU-only: a CUDA input faults in
# at::native::fbgemm_linear_quantize_weight, so the oracle and the injected
# candidate both receive CPU tensors -- that is the operator's real contract, not
# a device adaptation. There is no broadcast, scalar-operand or parameter axis.

_DTYPES = [torch.float32]

# The spec's 7-shape grid keeps only the ranks this operator accepts; the
# benchmark builder's weight-shaped blocks are included as well, so the two
# files exercise the same rank >= 2 contract.
_GRID_SHAPES = tu.selected_cases(
    [
        (2, 19, 7),
        (3, 5),
        (1, 8),
        (2, 1),
        (33, 129),
        (128, 127),
        (1024, 257),
        (65, 31, 5),
        (4, 6, 8, 10),
        (64, 64),
        (256, 256),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ],
    quick=[(2, 19, 7), (1, 8), (2, 1)],
)

# Row-block lengths on both sides of the vector widths a per-row quantizer may
# specialize on.
_ROW_LENGTHS = tu.selected_cases(
    [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 1023, 1024],
    quick=[
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        15,
        16,
        31,
        32,
        63,
        64,
        127,
        128,
        255,
        256,
        257,
        1023,
        1024,
    ],
)

# Constant rows take the degenerate min == max branch (a sentinel scale and
# zero_point, and a bias that cancels out).
_CONSTANT_ROWS = [
    [0.0, 0.0, 0.0, 0.0],
    [5.0, 5.0, 5.0, 5.0],
    [-5.0, -5.0, -5.0, -5.0],
    [1e-8, 1e-8, 1e-8, 1e-8],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, -1.0],
    [1e-2, -1e-2, 0.0, 7e-3],
    [-2.5, 2.5, 0.0, 0.0],
]

_ZERO_ROW_POSITIONS = [0, 1, 2]

# All three views are tiny, so they stay in the quick subset as well.
_LAYOUTS = ["transpose", "slice", "offset", "expanded"]

_EMPTY_SHAPES = [(0, 4)]

_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases([torch.float32]), quick=[])

# Probed: the kernel rejects every other dtype with
# "expected scalar type Float but found <dtype>".
_REJECTED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
]

# Probed: dim < 2 raises IndexError in the native kernel.
_REJECTED_SHAPES = [(), (1,), (256,)]


def _assert_quantized(res, ref):
    """Compare a (q, bias, scale, zero_point) result with the native tuple."""
    assert len(res) == 4, len(res)
    res_q, res_bias, res_scale, res_zp = res
    ref_q, ref_bias, ref_scale, ref_zp = ref

    # Schema types: the native operator returns a Python float and a Python int.
    assert isinstance(res_scale, float), type(res_scale)
    assert isinstance(res_zp, int) and not isinstance(res_zp, bool), type(res_zp)

    tu.assert_result_equal(res_q, ref_q)
    tu.assert_result_equal(res_bias, ref_bias)
    # Keep the returned double instead of narrowing to the input dtype, and allow
    # no absolute slack: the step size is divided out by the consumer.
    tu.assert_result_close(
        torch.tensor(res_scale, dtype=torch.float64),
        torch.tensor(ref_scale, dtype=torch.float64),
        atol=0,
    )
    assert res_zp == ref_zp, f"zero_point {res_zp} != {ref_zp}"


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _GRID_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_fbgemm_linear_quantize_weight_value_ranges(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range).cpu()
    ref_inp = tu.to_reference(inp)
    inp_before = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_linear_quantize_weight(ref_inp)
    res_out = flag_gems.fbgemm_linear_quantize_weight(inp)

    _assert_quantized(res_out, ref_out)
    # The weight is read, not written: only fresh buffers are returned.
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("row_length", _ROW_LENGTHS)
def test_fbgemm_linear_quantize_weight_row_length(row_length):
    inp = tu.make_input(torch.float32, (3, row_length), ["-1", "1"]).cpu()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_linear_quantize_weight(ref_inp)
    res_out = flag_gems.fbgemm_linear_quantize_weight(inp)

    _assert_quantized(res_out, ref_out)


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("row", _CONSTANT_ROWS)
def test_fbgemm_linear_quantize_weight_constant_row(row):
    inp = torch.tensor([row], dtype=torch.float32)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_linear_quantize_weight(ref_inp)
    res_out = flag_gems.fbgemm_linear_quantize_weight(inp)

    _assert_quantized(res_out, ref_out)


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("zero_row", _ZERO_ROW_POSITIONS)
def test_fbgemm_linear_quantize_weight_zero_row(zero_row):
    rows = [
        [1.5, -0.5, 2.0, -3.0],
        [0.25, 0.5, -1.0, 4.0],
        [-2.0, 1.0, 0.0, 0.5],
    ]
    rows[zero_row] = [0.0, 0.0, 0.0, 0.0]
    inp = torch.tensor(rows, dtype=torch.float32)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_linear_quantize_weight(ref_inp)
    res_out = flag_gems.fbgemm_linear_quantize_weight(inp)

    _assert_quantized(res_out, ref_out)


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_fbgemm_linear_quantize_weight_non_contiguous(layout):
    # Transposed, stride-2 and storage-offset views: the native kernel normalizes
    # the layout internally and returns what the contiguous input produces.
    block = tu.make_input(torch.float32, (6, 8), ["-1", "1"]).cpu()
    if layout == "transpose":
        inp = block.t()
    elif layout == "slice":
        inp = block[:, ::2]
    elif layout == "offset":
        inp = block[:, 1:5]
    else:
        inp = block[:1].expand(6, 8)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_linear_quantize_weight(ref_inp)
    res_out = flag_gems.fbgemm_linear_quantize_weight(inp)

    _assert_quantized(res_out, ref_out)


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("shape", _EMPTY_SHAPES)
def test_fbgemm_linear_quantize_weight_empty_input(shape):
    # No row block exists, but the returned tuple is still fully defined.
    inp = tu.make_input(torch.float32, shape, ["-1", "1"]).cpu()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_linear_quantize_weight(ref_inp)
    res_out = flag_gems.fbgemm_linear_quantize_weight(inp)

    _assert_quantized(res_out, ref_out)


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_fbgemm_linear_quantize_weight_special_values(dtype, scenario):
    # The shared payload is 1-D; the operator needs rank >= 2, so it becomes a
    # single row block.
    inp = tu.make_special_input(dtype, scenario).cpu().reshape(1, 5)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_linear_quantize_weight(ref_inp)
    res_out = flag_gems.fbgemm_linear_quantize_weight(inp)

    _assert_quantized(res_out, ref_out)


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test_fbgemm_linear_quantize_weight_rejects_unsupported_dtype(dtype):
    inp = tu.make_input(dtype, (4, 8), ["-1", "1"]).cpu()

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_quantize_weight(inp)


@pytest.mark.fbgemm_linear_quantize_weight
@pytest.mark.parametrize("shape", _REJECTED_SHAPES)
def test_fbgemm_linear_quantize_weight_rejects_rank_below_two(shape):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"]).cpu()

    with pytest.raises((IndexError, RuntimeError, ValueError)):
        flag_gems.fbgemm_linear_quantize_weight(inp)


@pytest.mark.fbgemm_linear_quantize_weight
def test_fbgemm_linear_quantize_weight_grad_metadata():
    inp = torch.randn((3, 8), dtype=torch.float32, device="cpu", requires_grad=True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_linear_quantize_weight(ref_inp)
    res_out = flag_gems.fbgemm_linear_quantize_weight(inp)

    _assert_quantized(res_out, ref_out)
    for res, ref in zip(res_out[:2], ref_out[:2]):
        assert res.requires_grad == ref.requires_grad
        assert (res.grad_fn is None) == (ref.grad_fn is None)
