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

"""Correctness tests for aten::mkldnn_max_pool3d_backward.

Native contract, probed with the exact operator: the three tensor operands are
MkldnnCPU, `output` must be a native mkldnn_max_pool3d result whose input
required grad (the backward re-creates that forward and consumes the argmax
workspace it saved), grad_output must match that forward output's shape, the
supported dtypes are float32 / bfloat16 / float16 and the operands are 5-D.
MkldnnCPU tensors exist on the CPU only, so the operands stay on the CPU and the
opaque oneDNN result is checked for its layout before its values are compared
through to_dense().
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

SUPPORTED_DTYPES = [torch.float32, torch.bfloat16, torch.float16]

# 5-D only: the native helper derives the pooling dimension from the kernel
# length and rejects any other rank (`expected dilation to be a single integer
# value or a list of 2 values to match the convolution dimensions`), so the
# spec's lower-rank shapes are exempt. The spec's 5-D entry (16, 7, 57, 32, 29)
# is kept and four smaller 5-D shapes are added.
DEFAULT_SHAPES = [
    (1, 1, 4, 4, 5),
    (1, 2, 8, 9, 10),
    (2, 3, 16, 17, 18),
    (2, 5, 9, 10, 11),
    (16, 7, 57, 32, 29),
]

# --quick adaptation of the required (2, 19, 7) shape to the 5-D contract.
QUICK_SHAPES = [(1, 2, 8, 9, 10)]

DEFAULT_PARAMS = {
    "kernel_size": [2, 2, 2],
    "stride": [2, 2, 2],
    "padding": [0, 0, 0],
    "dilation": [1, 1, 1],
    "ceil_mode": False,
}

# kernel_size / stride / padding are positive 3-vectors, dilation only works at
# 1 (the operand-building forward refuses anything else) and padding stays within
# half the kernel, the native pooling descriptor's bound. Rows that omit padding /
# dilation / ceil_mode exercise those schema defaults by omission; the stride
# default `[]` is not native-valid for 5-D and is a negative case instead. stride
# 0 is never passed: it crashes the vendor kernel with SIGFPE.
PARAM_ROWS = [
    {"kernel_size": [2, 2, 2], "stride": [2, 2, 2]},
    {"kernel_size": [2, 2, 2], "stride": [2, 2, 2], "padding": [0, 0, 0]},
    {"kernel_size": [1, 1, 1], "stride": [1, 1, 1]},
    {"kernel_size": [3, 3, 3], "stride": [1, 1, 1]},
    {"kernel_size": [2, 3, 4], "stride": [2, 3, 4]},
    {"kernel_size": [2, 2, 2], "stride": [2, 2, 2], "padding": [1, 1, 1]},
    {
        "kernel_size": [3, 3, 3],
        "stride": [3, 3, 3],
        "padding": [1, 1, 1],
        "ceil_mode": True,
    },
    {
        "kernel_size": [3, 3, 3],
        "stride": [3, 3, 3],
        "dilation": [1, 1, 1],
        "ceil_mode": False,
    },
]

# All eight rows are verified native-valid on this cheap 5-D descriptor, so every
# parameter branch is a quick case instead of being dropped.
PARAM_DESCRIPTOR = (1, 2, 8, 9, 10)
QUICK_PARAM_ROWS = list(PARAM_ROWS)

# Singleton and zero-extent boundaries. A zero-sized extent is a valid oneDNN
# memory descriptor, but only for kernel extents that fit inside it; rows 2, 5 and
# 6 (1x1x1/1, 2x2x2 stride 2 padding 1, 3x3x3 stride 3 padding 1) are the rows
# verified for all of these shapes, whereas a larger kernel fails with `could not
# construct a memory descriptor using a format tag`.
BOUNDARY_SHAPES = [
    (1, 1, 1, 1, 1),
    (2, 3, 1, 1, 1),
    (0, 1, 4, 4, 4),
    (1, 0, 4, 4, 4),
    (1, 1, 4, 4, 0),
]
BOUNDARY_PARAM_ROWS = [PARAM_ROWS[2], PARAM_ROWS[5], PARAM_ROWS[6]]

# Invalid input is rejected by the backend (RuntimeError) or earlier by a
# candidate that validates its arguments (TypeError).
INVALID_INPUT_ERRORS = (RuntimeError, TypeError)

# nan / inf / mixed per supported dtype, in grad_output (the operand the backward
# scatters) and separately in the forward input. Only these three floating dtypes
# can be MkldnnCPU operands, so there is no float8 scenario.
SPECIAL_CASES = tu.selected_cases(
    [
        (where, dtype, scenario)
        for where in ("grad", "input")
        for dtype, scenario in tu.special_value_cases(SUPPORTED_DTYPES)
    ],
    quick=[],
)


def _cpu_input(dtype, shape, value_range):
    """Build a value-range tensor for this operator's CPU-only contract."""
    # tu.make_input builds on flag_gems.device; MkldnnCPU operands exist only on
    # the CPU, so the tensor is transferred before it becomes an operand.
    return tu.make_input(dtype, shape, value_range).cpu()


def _special_dense(dtype, scenario, shape):
    """Tile the shared special-value payload into a dense CPU tensor."""
    payload = tu.make_special_input(dtype, scenario).cpu()
    count = 1
    for size in shape:
        count *= size
    repeats = -(-count // payload.numel())
    return payload.repeat(repeats)[:count].reshape(shape)


def _operands(dense, params):
    """Run the exact native forward and return its (output, input) mkldnn pair.

    `output` is returned untouched: it carries the argmax workspace the backward
    needs, so cloning or rebuilding it would lose that state.
    """
    x = dense.detach().clone().requires_grad_(True)
    output = torch.ops.aten.mkldnn_max_pool3d(x.to_mkldnn(), **params)
    return output, x.to_mkldnn()


@pytest.mark.mkldnn_max_pool3d_backward
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_cases(DEFAULT_SHAPES, quick=QUICK_SHAPES))
def test_mkldnn_max_pool3d_backward_value_range(shape, value_range, dtype):
    inp = _cpu_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    ref_output, ref_input = _operands(ref_inp, DEFAULT_PARAMS)
    grad = _cpu_input(dtype, tuple(ref_output.shape), value_range)
    ref_grad = tu.to_reference(grad)
    res_output, res_input = _operands(inp, DEFAULT_PARAMS)

    ref_out = torch.ops.aten.mkldnn_max_pool3d_backward(
        ref_grad.to_mkldnn(), ref_output, ref_input, **DEFAULT_PARAMS
    )
    res_out = flag_gems.mkldnn_max_pool3d_backward(
        grad.to_mkldnn(), res_output, res_input, **DEFAULT_PARAMS
    )

    # Scatter/indexing result: the values are copied, never rounded, so they are
    # compared at the shared tolerance after the layout check.
    assert res_out.is_mkldnn
    tu.assert_result_close(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_max_pool3d_backward
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "params", tu.selected_cases(PARAM_ROWS, quick=QUICK_PARAM_ROWS)
)
def test_mkldnn_max_pool3d_backward_params(params, dtype):
    inp = _cpu_input(dtype, PARAM_DESCRIPTOR, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_output, ref_input = _operands(ref_inp, params)
    grad = _cpu_input(dtype, tuple(ref_output.shape), ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    res_output, res_input = _operands(inp, params)

    ref_out = torch.ops.aten.mkldnn_max_pool3d_backward(
        ref_grad.to_mkldnn(), ref_output, ref_input, **params
    )
    res_out = flag_gems.mkldnn_max_pool3d_backward(
        grad.to_mkldnn(), res_output, res_input, **params
    )

    assert res_out.is_mkldnn
    tu.assert_result_close(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_max_pool3d_backward
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "params", tu.selected_cases(BOUNDARY_PARAM_ROWS, quick=BOUNDARY_PARAM_ROWS)
)
@pytest.mark.parametrize(
    "shape", tu.selected_cases(BOUNDARY_SHAPES, quick=BOUNDARY_SHAPES)
)
def test_mkldnn_max_pool3d_backward_boundaries(shape, params, dtype):
    # Metadata-preserving boundaries: a zero extent still has to come back with
    # the operand's shape, dtype and oneDNN layout rather than failing or
    # collapsing to a non-empty tensor.
    inp = _cpu_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_output, ref_input = _operands(ref_inp, params)
    grad = _cpu_input(dtype, tuple(ref_output.shape), ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    res_output, res_input = _operands(inp, params)

    ref_out = torch.ops.aten.mkldnn_max_pool3d_backward(
        ref_grad.to_mkldnn(), ref_output, ref_input, **params
    )
    res_out = flag_gems.mkldnn_max_pool3d_backward(
        grad.to_mkldnn(), res_output, res_input, **params
    )

    assert res_out.is_mkldnn
    assert tuple(res_out.to_dense().shape) == tuple(shape)
    tu.assert_result_close(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_max_pool3d_backward
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_max_pool3d_backward_out(dtype):
    # The native .out overload is callable with an MkldnnCPU out buffer and
    # returns that very buffer, so the candidate is asked for the same form.
    inp = _cpu_input(dtype, PARAM_DESCRIPTOR, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_output, ref_input = _operands(ref_inp, DEFAULT_PARAMS)
    grad = _cpu_input(dtype, tuple(ref_output.shape), ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    res_output, res_input = _operands(inp, DEFAULT_PARAMS)

    ref_buf = torch.empty(PARAM_DESCRIPTOR, dtype=dtype).to_mkldnn()
    ref_out = torch.ops.aten.mkldnn_max_pool3d_backward.out(
        ref_grad.to_mkldnn(),
        ref_output,
        ref_input,
        out=ref_buf,
        **DEFAULT_PARAMS,
    )
    res_buf = torch.empty(PARAM_DESCRIPTOR, dtype=dtype).to_mkldnn()
    res_out = flag_gems.mkldnn_max_pool3d_backward(
        grad.to_mkldnn(),
        res_output,
        res_input,
        out=res_buf,
        **DEFAULT_PARAMS,
    )

    assert res_out.is_mkldnn
    assert res_out is res_buf
    tu.assert_result_close(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_max_pool3d_backward
@pytest.mark.parametrize("where,dtype,scenario", SPECIAL_CASES)
def test_mkldnn_max_pool3d_backward_special_values(where, dtype, scenario):
    # The backward scatters grad_output through the indices the forward saved, so
    # a NaN/Inf grad_output propagates while a NaN/Inf forward input only fills a
    # window value; both placements are covered explicitly.
    if where == "input":
        inp = _special_dense(dtype, scenario, PARAM_DESCRIPTOR)
    else:
        inp = _cpu_input(dtype, PARAM_DESCRIPTOR, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_output, ref_input = _operands(ref_inp, DEFAULT_PARAMS)
    if where == "grad":
        grad = _special_dense(dtype, scenario, tuple(ref_output.shape))
    else:
        grad = _cpu_input(dtype, tuple(ref_output.shape), ["-1", "1"])
    ref_grad = tu.to_reference(grad)
    res_output, res_input = _operands(inp, DEFAULT_PARAMS)

    ref_out = torch.ops.aten.mkldnn_max_pool3d_backward(
        ref_grad.to_mkldnn(), ref_output, ref_input, **DEFAULT_PARAMS
    )
    res_out = flag_gems.mkldnn_max_pool3d_backward(
        grad.to_mkldnn(), res_output, res_input, **DEFAULT_PARAMS
    )

    assert res_out.is_mkldnn
    tu.assert_result_close(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_max_pool3d_backward
def test_mkldnn_max_pool3d_backward_keeps_operands_unchanged():
    dtype = torch.float32
    inp = _cpu_input(dtype, PARAM_DESCRIPTOR, ["-1", "1"])
    res_output, res_input = _operands(inp, DEFAULT_PARAMS)
    # grad_output has to match the forward output's shape; a gradient shaped like
    # the *input* cannot be reordered into the pooling layout and the backend then
    # raises `could not create a primitive descriptor for the reorder primitive`.
    grad = _cpu_input(dtype, tuple(res_output.shape), ["-1", "1"])
    grad_mkldnn = grad.to_mkldnn()
    grad_before = tu.to_reference(grad)
    output_before = tu.to_reference(res_output.to_dense())
    input_before = tu.to_reference(inp)

    res_out = flag_gems.mkldnn_max_pool3d_backward(
        grad_mkldnn, res_output, res_input, **DEFAULT_PARAMS
    )

    # Out-of-place: a fresh MkldnnCPU tensor whose three operands all still hold
    # the values snapshotted before the call.
    assert res_out.is_mkldnn
    assert res_out is not grad_mkldnn
    assert res_out is not res_output
    assert res_out is not res_input
    tu.assert_result_equal(grad_mkldnn.to_dense(), grad_before)
    tu.assert_result_equal(res_output.to_dense(), output_before)
    tu.assert_result_equal(res_input.to_dense(), input_before)


@pytest.mark.mkldnn_max_pool3d_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_mkldnn_max_pool3d_backward_rejects_strided_operands(dtype):
    # Strided operands are never valid (the native op raises `itensor_from_mkldnn
    # expects MKL-DNN tensor input`). float8 is included because it cannot be an
    # mkldnn operand at all (`dense_to_mkldnn expects float, bfloat16, half,
    # uint8, int8 tensor input`).
    inp = _cpu_input(dtype, PARAM_DESCRIPTOR, ["-1", "1"])
    grad = _cpu_input(dtype, PARAM_DESCRIPTOR, ["-1", "1"])

    with pytest.raises(INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_max_pool3d_backward(grad, inp, inp, **DEFAULT_PARAMS)


@pytest.mark.mkldnn_max_pool3d_backward
def test_mkldnn_max_pool3d_backward_rejects_strided_grad_output():
    # The mkldnn requirement is checked per operand: here only grad_output is
    # strided while output and input are valid MkldnnCPU tensors.
    inp = _cpu_input(torch.float32, PARAM_DESCRIPTOR, ["-1", "1"])
    output, input_mkldnn = _operands(inp, DEFAULT_PARAMS)
    grad = _cpu_input(torch.float32, tuple(output.shape), ["-1", "1"])

    with pytest.raises(INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_max_pool3d_backward(
            grad, output, input_mkldnn, **DEFAULT_PARAMS
        )


@pytest.mark.mkldnn_max_pool3d_backward
def test_mkldnn_max_pool3d_backward_rejects_2d_pooling_descriptor():
    # A 2-length kernel on 5-D operands: the native helper demands `a single
    # integer value or a list of 3 values`.
    inp = _cpu_input(torch.float32, PARAM_DESCRIPTOR, ["-1", "1"])
    output, input_mkldnn = _operands(inp, DEFAULT_PARAMS)
    grad = _cpu_input(torch.float32, tuple(output.shape), ["-1", "1"])

    with pytest.raises(INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_max_pool3d_backward(
            grad.to_mkldnn(), output, input_mkldnn, [2, 2], [2, 2, 2]
        )


@pytest.mark.mkldnn_max_pool3d_backward
def test_mkldnn_max_pool3d_backward_rejects_negative_padding():
    # Negative padding cannot build a pooling descriptor (`could not create a
    # descriptor for a pooling forward propagation primitive`).
    inp = _cpu_input(torch.float32, PARAM_DESCRIPTOR, ["-1", "1"])
    output, input_mkldnn = _operands(inp, DEFAULT_PARAMS)
    grad = _cpu_input(torch.float32, tuple(output.shape), ["-1", "1"])
    params = dict(DEFAULT_PARAMS, padding=[-1, -1, -1])

    with pytest.raises(INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_max_pool3d_backward(
            grad.to_mkldnn(), output, input_mkldnn, **params
        )


@pytest.mark.mkldnn_max_pool3d_backward
def test_mkldnn_max_pool3d_backward_rejects_missing_stride():
    # The schema default for stride is an empty list, which is not native-valid
    # for a 5-D operand (`got stride=[]`), so every workload above passes stride
    # explicitly and the omission is covered here instead. The stride must also be
    # positive: 0 SIGFPEs the vendor kernel.
    inp = _cpu_input(torch.float32, PARAM_DESCRIPTOR, ["-1", "1"])
    output, input_mkldnn = _operands(inp, DEFAULT_PARAMS)
    grad = _cpu_input(torch.float32, tuple(output.shape), ["-1", "1"])

    with pytest.raises(INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_max_pool3d_backward(
            grad.to_mkldnn(), output, input_mkldnn, [2, 2, 2]
        )


@pytest.mark.mkldnn_max_pool3d_backward
def test_mkldnn_max_pool3d_backward_rejects_forward_without_workspace():
    # A forward whose input did not require grad saves no argmax workspace and the
    # native backward then fails with `could not execute a primitive`.
    inp = _cpu_input(torch.float32, PARAM_DESCRIPTOR, ["-1", "1"])
    input_mkldnn = inp.to_mkldnn()
    output = torch.ops.aten.mkldnn_max_pool3d(input_mkldnn, **DEFAULT_PARAMS)
    grad = _cpu_input(torch.float32, tuple(output.shape), ["-1", "1"])

    with pytest.raises(INVALID_INPUT_ERRORS):
        flag_gems.mkldnn_max_pool3d_backward(
            grad.to_mkldnn(), output, input_mkldnn, **DEFAULT_PARAMS
        )
