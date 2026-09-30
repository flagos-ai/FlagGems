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

"""Correctness tests for the MkldnnCPU-only ``mkldnn_max_pool2d_backward``.

The operator dispatches MkldnnCPU only: the native reference and the injected
candidate both receive opaque ``torch._mkldnn`` CPU tensors, and the shared
value assertions run on their materialized contents after the candidate's
layout and device have been checked.

A backward only runs against the oneDNN pooling workspace that
``mkldnn_max_pool2d`` stores when grad mode is on and its input requires grad,
so every workload builds its own grad-enabled training forward.

Two spec dimensions have no applicable workload here, each for a native reason:
there is no broadcast form (the three tensor arguments share one pooling
layout) and no second-order gradient (the native backward has no autograd
kernel).

2-D pooling reads (N, C, H, W) only, so the spec's rank ladder is mapped onto
4-D: the 4-D entry is kept verbatim, the 5-D entry folds to (112, 57, 32, 29)
with the same element count, and the 0/1/2/3-D entries (no two-dimensional
spatial extent) become the smallest usable 4-D shapes.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

_SHAPES = [
    (2, 19, 7, 7),
    (1, 1, 1024, 1024),
    (1, 20, 320, 15),
    (16, 128, 64, 60),
    (112, 57, 32, 29),
    (2, 3, 9, 9),
    (8, 16, 32, 32),
    (2, 2, 5, 5),
]

# The spec's quick shape (2, 19, 7) is rank 3, lifted here to a 4-D pooling input.
_SHAPE_CASES = tu.selected_cases(_SHAPES, quick=[(2, 19, 7, 7)])

# dense_to_mkldnn converts float, bfloat16, half, uint8 and int8; oneDNN rejects
# the uint8/int8 pooling descriptor, and float64/int32/int64/bool have no mkldnn
# representation at all, so these three are the supported dtypes.
_DTYPES = [torch.float32, torch.bfloat16, torch.float16]

_CONFIG = ([2, 2], [2, 2], [0, 0], [1, 1], False)

# (input shape, config) rows with native-valid static geometries: the window
# fits the extent, padding stays at or below half the kernel, and the training
# forward (the pooling itself) requires dilation == 1, so the forward always
# uses [1, 1].  The rows are small so the quick level still covers every
# parameter branch: stride equal to and greater than the kernel, padding,
# overlapping windows, an asymmetric kernel and stride, ceil_mode, 1x1, and
# empty inputs.
_GEOMETRIES = [
    ((1, 1, 1, 1), ([1, 1], [1, 1], [0, 0], [1, 1], False)),
    ((2, 3, 9, 9), ([5, 5], [1, 1], [0, 0], [1, 1], False)),
    ((2, 3, 9, 9), ([9, 9], [1, 1], [0, 0], [1, 1], False)),
    ((2, 3, 28, 28), ([2, 2], [2, 2], [0, 0], [1, 1], False)),
    ((2, 3, 29, 29), ([2, 2], [2, 2], [0, 0], [1, 1], True)),
    ((2, 3, 32, 32), ([3, 3], [2, 2], [1, 1], [1, 1], False)),
    ((2, 3, 32, 32), ([3, 3], [2, 2], [1, 1], [1, 1], True)),
    ((2, 3, 56, 15), ([2, 3], [1, 2], [0, 0], [1, 1], False)),
    ((2, 3, 7, 7), ([2, 2], [2, 2], [1, 1], [1, 1], False)),
    ((2, 3, 8, 8), ([2, 2], [3, 3], [0, 0], [1, 1], True)),
    ((2, 3, 9, 9), ([3, 3], [1, 1], [1, 1], [1, 1], False)),
    ((2, 3, 57, 32), ([1, 1], [1, 1], [0, 0], [1, 1], False)),
    ((0, 3, 9, 9), ([2, 2], [2, 2], [0, 0], [1, 1], False)),
    ((2, 3, 0, 9), ([2, 2], [2, 2], [0, 0], [1, 1], False)),
    ((2, 3, 9, 0), ([2, 2], [2, 2], [0, 0], [1, 1], False)),
]

# The same geometries once on the folded 5-D spec shape, so the default level
# also carries a performance-relevant extent; the quick level keeps the small
# rows above.
_BIG_SHAPE = (112, 57, 32, 29)
_BIG_GEOMETRIES = [
    (_BIG_SHAPE, ([2, 2], [2, 2], [0, 0], [1, 1], False)),
    (_BIG_SHAPE, ([2, 2], [2, 2], [0, 0], [1, 1], True)),
    (_BIG_SHAPE, ([3, 3], [2, 2], [1, 1], [1, 1], True)),
    (_BIG_SHAPE, ([3, 3], [1, 1], [1, 1], [1, 1], False)),
]

_GEOMETRY_CASES = tu.selected_cases(_GEOMETRIES + _BIG_GEOMETRIES, quick=_GEOMETRIES)

# A gradInput has the pooling-input shape and oneDNN cannot resize an out
# buffer, so the out workload keeps the non-empty geometries.
_OUT_CASES = [row for row in _GEOMETRY_CASES if 0 not in row[0]]

# A 1x1 window makes every position its own window, so a payload placed in
# grad_output reaches the output unchanged and nan/inf patterns stay observable.
_SPECIAL_CONFIG = ([1, 1], [1, 1], [0, 0], [1, 1], False)
_SPECIAL_SHAPE = (1, 1, 1, 5)
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])

_GRAD_CASES = tu.selected_cases(
    [
        (torch.float32, (2, 3, 9, 9)),
        (torch.bfloat16, (2, 3, 9, 9)),
        (torch.float16, (2, 3, 9, 9)),
        (torch.float32, (2, 3, 29, 29)),
    ],
    quick=[],
)

_NEG_SHAPE = (2, 3, 9, 9)
_ARG_POSITIONS = {"grad_output": 0, "output": 1, "input": 2}


def _cpu_input(dtype, shape, value_range):
    # Spec value-range input on the CPU, the only device this op dispatches on.
    return tu.make_input(dtype, shape, value_range).to("cpu")


def _training_forward(dense, config):
    # mkldnn_max_pool2d keeps the oneDNN pooling workspace only under grad mode,
    # and the backward reads that workspace from the returned output.
    kernel_size, stride, padding, dilation, ceil_mode = config
    x = dense.to_mkldnn().requires_grad_(True)
    with torch.enable_grad():
        output = torch.ops.aten.mkldnn_max_pool2d(
            x, kernel_size, stride, padding, dilation, ceil_mode
        )
    return x.detach(), output


def _mkldnn_args(dtype, shape, value_range, config):
    # Independent (grad_output, output, input) mkldnn triples for both paths.
    # They share values but no storage, and each path runs its own training
    # forward because the workspace lives inside the output tensor.
    inp = _cpu_input(dtype, shape, value_range)
    ref_x, ref_output = _training_forward(tu.to_reference(inp), config)
    grad = _cpu_input(dtype, tuple(ref_output.shape), ["-1", "1"])
    res_x, res_output = _training_forward(inp, config)
    return (
        (tu.to_reference(grad).to_mkldnn(), ref_output, ref_x),
        (grad.to_mkldnn(), res_output, res_x),
    )


def _valid_args(dtype=torch.float32, shape=_NEG_SHAPE, config=_CONFIG):
    # One valid candidate argument triple, to isolate a single bad input.
    return _mkldnn_args(dtype, shape, ["-1", "1"], config)[1]


def _assert_mkldnn_result(res_out, ref_out):
    # The candidate must return the opaque CPU pooling layout; the shared value
    # assertions only see materialized contents, so check it before to_dense.
    assert res_out.layout == torch._mkldnn
    assert res_out.device.type == "cpu"
    tu.assert_result_close(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", _SHAPE_CASES)
def test_mkldnn_max_pool2d_backward(shape, value_range, dtype):
    ref_args, res_args = _mkldnn_args(dtype, shape, value_range, _CONFIG)

    ref_out = torch.ops.aten.mkldnn_max_pool2d_backward(*ref_args, *_CONFIG)
    res_out = flag_gems.mkldnn_max_pool2d_backward(*res_args, *_CONFIG)

    _assert_mkldnn_result(res_out, ref_out)
    for operand, reference in zip(res_args, ref_args):
        tu.assert_result_equal(operand.to_dense(), reference.to_dense())


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape,config", _GEOMETRY_CASES)
def test_mkldnn_max_pool2d_backward_geometries(shape, config, dtype):
    ref_args, res_args = _mkldnn_args(dtype, shape, ["-1", "1"], config)

    ref_out = torch.ops.aten.mkldnn_max_pool2d_backward(*ref_args, *config)
    res_out = flag_gems.mkldnn_max_pool2d_backward(*res_args, *config)

    _assert_mkldnn_result(res_out, ref_out)


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape,config", _OUT_CASES)
def test_mkldnn_max_pool2d_backward_out(shape, config, dtype):
    kernel_size, stride, padding, dilation, ceil_mode = config
    ref_args, res_args = _mkldnn_args(dtype, shape, ["-1", "1"], config)
    # The out buffer must be an input-shaped mkldnn tensor of the input dtype;
    # other shapes or dtypes are rejected by the native operator.
    ref_buf = torch.zeros(shape, dtype=dtype).to_mkldnn()
    res_buf = torch.zeros(shape, dtype=dtype).to_mkldnn()

    torch.ops.aten.mkldnn_max_pool2d_backward.out(
        *ref_args, kernel_size, stride, padding, dilation, ceil_mode, out=ref_buf
    )
    res_ret = flag_gems.mkldnn_max_pool2d_backward(
        *res_args, kernel_size, stride, padding, dilation, ceil_mode, out=res_buf
    )

    assert res_ret is res_buf
    _assert_mkldnn_result(res_buf, ref_buf)


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_max_pool2d_backward_default_arguments(dtype):
    # padding, dilation and ceil_mode are omitted here, so the schema defaults
    # ([0, 0], [1, 1], False) must be honoured; the training forward uses the
    # same defaults.
    ref_args, res_args = _mkldnn_args(dtype, _NEG_SHAPE, ["-1", "1"], _CONFIG)

    ref_out = torch.ops.aten.mkldnn_max_pool2d_backward(
        grad_output=ref_args[0],
        output=ref_args[1],
        input=ref_args[2],
        kernel_size=[2, 2],
        stride=[2, 2],
    )
    res_out = flag_gems.mkldnn_max_pool2d_backward(
        grad_output=res_args[0],
        output=res_args[1],
        input=res_args[2],
        kernel_size=[2, 2],
        stride=[2, 2],
    )

    _assert_mkldnn_result(res_out, ref_out)


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_mkldnn_max_pool2d_backward_special_values(dtype, scenario):
    dense = _cpu_input(dtype, _SPECIAL_SHAPE, ["-1", "1"])
    payload = tu.make_special_input(dtype, scenario).to("cpu")
    ref_x, ref_output = _training_forward(tu.to_reference(dense), _SPECIAL_CONFIG)
    res_x, res_output = _training_forward(dense, _SPECIAL_CONFIG)

    ref_out = torch.ops.aten.mkldnn_max_pool2d_backward(
        tu.to_reference(payload.reshape(tuple(ref_output.shape))).to_mkldnn(),
        ref_output,
        ref_x,
        *_SPECIAL_CONFIG,
    )
    res_out = flag_gems.mkldnn_max_pool2d_backward(
        payload.reshape(tuple(res_output.shape)).to_mkldnn(),
        res_output,
        res_x,
        *_SPECIAL_CONFIG,
    )

    _assert_mkldnn_result(res_out, ref_out)


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize("dtype,shape", _GRAD_CASES)
def test_mkldnn_max_pool2d_backward_gradient(dtype, shape):
    # The reference gradient comes from autograd through the original mkldnn
    # leaf, which routes into this same native backward; the candidate must
    # reproduce it for a nonuniform upstream gradient.
    inp = _cpu_input(dtype, shape, ["-1", "1"])
    ref_x = tu.to_reference(inp).to_mkldnn().requires_grad_(True)
    res_x = inp.to_mkldnn().requires_grad_(True)
    with torch.enable_grad():
        ref_output = torch.ops.aten.mkldnn_max_pool2d(ref_x, *_CONFIG)
        res_output = torch.ops.aten.mkldnn_max_pool2d(res_x, *_CONFIG)
    grad = _cpu_input(dtype, tuple(ref_output.shape), ["-1", "1"])

    (ref_grad,) = torch.autograd.grad(
        (ref_output,), (ref_x,), grad_outputs=(tu.to_reference(grad).to_mkldnn(),)
    )
    res_grad = flag_gems.mkldnn_max_pool2d_backward(
        grad.to_mkldnn(), res_output, res_x.detach(), *_CONFIG
    )

    _assert_mkldnn_result(res_grad, ref_grad)


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize("bad_arg", ["grad_output", "output", "input", "absent_output"])
def test_mkldnn_max_pool2d_backward_rejects_strided_arguments(bad_arg):
    # MkldnnCPU only: a dense or absent tensor argument cannot be read
    # ('itensor_from_mkldnn expects MKL-DNN tensor input').
    args = list(_valid_args())
    if bad_arg == "absent_output":
        args[1] = None
    else:
        args[_ARG_POSITIONS[bad_arg]] = args[_ARG_POSITIONS[bad_arg]].to_dense()

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.mkldnn_max_pool2d_backward(*args, *_CONFIG)


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize(
    "kernel_size,stride,padding",
    [
        pytest.param([0, 0], [2, 2], [0, 0], id="kernel_size_zero"),
        pytest.param([2, 2], [0, 0], [0, 0], id="stride_zero"),
        pytest.param([9, 9], [1, 1], [0, 0], id="kernel_size_larger_than_input"),
        pytest.param([5, 5], [1, 1], [0, 0], id="kernel_size_5_on_9"),
        pytest.param([2, 2], [2, 2], [2, 2], id="padding_exceeds_half_kernel"),
        pytest.param([2, 2], [1, 1], [1, 1], id="padding_1_with_kernel_2"),
        pytest.param([-1, -1], [2, 2], [0, 0], id="negative_kernel_size"),
        pytest.param([2, 2], [-1, -1], [0, 0], id="negative_stride"),
        pytest.param([2, 2], [2, 2], [-1, -1], id="negative_padding"),
    ],
)
def test_mkldnn_max_pool2d_backward_rejects_incompatible_pooling_params(
    kernel_size, stride, padding
):
    # These parameters are invalid or disagree with the workspace produced
    # by the fixed _CONFIG forward; valid larger kernels are covered separately.
    grad, output, x = _valid_args()

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.mkldnn_max_pool2d_backward(
            grad, output, x, kernel_size, stride, padding, [1, 1], False
        )


@pytest.mark.mkldnn_max_pool2d_backward
def test_mkldnn_max_pool2d_backward_rejects_omitted_stride():
    # The schema default for stride is [], which the native operator rejects
    # ('expected stride to be a single integer value or a list of 2 values').
    grad, output, x = _valid_args()

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.mkldnn_max_pool2d_backward(
            grad_output=grad,
            output=output,
            input=x,
            kernel_size=[2, 2],
            padding=[0, 0],
        )


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
def test_mkldnn_max_pool2d_backward_rejects_integer_dtypes(dtype):
    # int8/uint8 do convert to mkldnn, but the pooling descriptor needs a
    # floating-point workspace and oneDNN rejects the integer geometry.
    inp = _cpu_input(dtype, _NEG_SHAPE, ["0", "1"]).to_mkldnn()

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.mkldnn_max_pool2d_backward(inp, inp, inp, *_CONFIG)


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ],
)
def test_mkldnn_max_pool2d_backward_rejects_unconvertible_dtypes(dtype):
    # float64/int32/int64/bool have no mkldnn representation
    # ('dense_to_mkldnn expects float, bfloat16, half, uint8, int8 tensor
    # input'), so the arguments stay strided and MkldnnCPU dispatch is
    # unreachable.
    grad, output, x = _valid_args()
    args = [t.to_dense().to(dtype) for t in (grad, output, x)]

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.mkldnn_max_pool2d_backward(*args, *_CONFIG)


@pytest.mark.mkldnn_max_pool2d_backward
@pytest.mark.parametrize("shape", [(2, 3, 9), (2, 3, 9, 9, 9)])
def test_mkldnn_max_pool2d_backward_rejects_non_4d_inputs(shape):
    # mkldnn holds rank 3 and rank 5 tensors, but 2-D pooling reads (N, C, H, W)
    # and the kernel_size length must match the rank.
    inp = tu.make_input(torch.float32, shape, ["-1", "1"]).to("cpu").to_mkldnn()

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.mkldnn_max_pool2d_backward(inp, inp, inp, *_CONFIG)


@pytest.mark.mkldnn_max_pool2d_backward
def test_mkldnn_max_pool2d_backward_out_rejects_dense_buffer():
    # Writing into a dense buffer would need a dense/mkldnn copy
    # ('copy_mkldnn_: between mkldnn layout and dense Tensors is not
    # implemented').
    grad, output, x = _valid_args()

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.mkldnn_max_pool2d_backward(
            grad, output, x, *_CONFIG, out=torch.zeros(_NEG_SHAPE)
        )
