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

# aten::_nnpack_spatial_convolution is the NNPACK CPU engine: it takes float32
# NCHW CPU tensors only and rejects accelerator tensors and every other dtype
# with "Mismatched Tensor types in NNPack convolutionOutput", so operands and the
# reference are built on the CPU. The build flag below is
# metadata only; initialization is deferred to the execution fixture.
_CPU = torch.device("cpu")
_NNPACK_AVAILABLE = "USE_NNPACK=ON" in torch.__config__.show()


@pytest.fixture(scope="module", autouse=True)
def initialize_nnpack():
    # Initialize the native engine during execution, never during collection.
    # No runtime skip: execution errors remain failures.
    torch.backends.nnpack.is_available()


pytestmark = [
    pytest.mark.nnpack_spatial_convolution,
    pytest.mark.skipif(
        not _NNPACK_AVAILABLE,
        reason="PyTorch was built without NNPACK",
    ),
]

# NNPACK is a fixed-rank 4-D NCHW engine, so the shared shape grid is expressed
# with 4-D rows only: (input shape, out_channels).
_CONV_SHAPES = [
    ((2, 3, 19, 7), 6),
    ((1, 1, 8, 8), 1),
    ((2, 4, 9, 7), 8),
    ((1, 16, 32, 32), 8),
    ((4, 8, 17, 15), 5),
    ((3, 5, 19, 17), 10),
    ((8, 4, 28, 28), 4),
    ((2, 7, 11, 13), 9),
    ((16, 128, 64, 60), 8),
]
_QUICK_CONV_SHAPES = [((2, 3, 19, 7), 6)]
_SHAPE_CASES = tu.selected_cases(_CONV_SHAPES, quick=_QUICK_CONV_SHAPES)

# Kernel sizes and the symmetric padding each one needs.
_KERNEL_CASES = [
    (1, [0, 0]),
    (2, [0, 0]),
    (3, [1, 1]),
    (5, [2, 2]),
]

# Padding/stride combinations, including asymmetric padding and stride 2.
_PAD_STRIDE_CASES = [
    ([1, 1], [1, 1]),
    ([0, 0], [1, 1]),
    ([2, 2], [1, 1]),
    ([1, 0], [1, 2]),
    ([1, 1], [2, 2]),
]

# (shape, out_channels, padding, stride) for gradient checks; default only.
_BACKWARD_CASES = tu.selected_cases(
    [
        ((2, 3, 19, 7), 6, [1, 1], [1, 1]),
        ((2, 4, 9, 7), 8, [1, 0], [1, 2]),
        ((1, 1, 8, 8), 1, [0, 0], [1, 1]),
        ((3, 5, 17, 15), 4, [2, 2], [2, 2]),
    ],
    quick=[],
)

# (kernel, padding, input plane, payload length) for the nonfinite matrix. The
# ones kernel makes each output the plain window sum, so nan/inf propagation is
# the convolution's own and needs no mask or tolerance change. Wider windows were
# measured to saturate whole outputs to NaN; these two geometries keep the
# comparison informative. Default only.
_SPECIAL_ROWS = [
    (1, [0, 0], (1, 1, 1, 5), 5),
    (3, [1, 1], (1, 1, 2, 2), 4),
]

# Every dtype the native kernel rejects on this backend, verified with a real
# call; NNPACK implements float32 only.
_UNSUPPORTED_DTYPES = [
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
]

# Rank 2/3 violate the 4-D precondition and an empty batch reaches the engine,
# which reports the update failure. Ranks above 4 are absent: the native shape
# precheck was observed to abort the process with SIGFPE (integer divide by zero)
# for a 6-D input instead of raising, so that negative cannot run in-process.
_INVALID_INPUT_SHAPES = [(8, 8), (3, 8, 8), (0, 3, 8, 8)]
# The non-square kernel weight (6, 3, 3, 5) is native-valid and is deliberately
# not listed here.
_BAD_WEIGHT_SHAPES = [(4, 3, 3), (4, 5, 3, 3), (0, 3, 3, 3)]
_BAD_PADDING = [[-1, 0], [-1, -1]]
# Negative strides only: a zero stride divides by zero inside the engine and
# terminates the process, so it is never invoked from a test.
_BAD_STRIDE = [[-1, 1], [1, -2]]


def _cpu_input(shape, value_range):
    """float32 CPU tensor carrying the shared value-range semantics.

    tu.make_input allocates on flag_gems.device, but the NNPACK engine rejects
    accelerator operands, so the same per-range bounds are materialized on the
    CPU here.
    """
    low = tu.resolve_bound(value_range[0], torch.float32)
    high = tu.resolve_bound(value_range[1], torch.float32)
    if low == high:
        return torch.full(shape, low, dtype=torch.float32, device=_CPU)
    return torch.testing.make_tensor(
        shape, dtype=torch.float32, device=_CPU, low=low, high=high
    )


def _conv_operands(shape, out_channels, value_range, kernel=3):
    inp = _cpu_input(shape, value_range)
    weight = _cpu_input((out_channels, shape[1], kernel, kernel), value_range)
    bias = _cpu_input((out_channels,), value_range)
    return inp, weight, bias


@pytest.mark.parametrize("shape,out_channels", _SHAPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__nnpack_spatial_convolution(shape, out_channels, value_range):
    inp, weight, bias = _conv_operands(shape, out_channels, value_range)
    ref_inp, ref_weight, ref_bias = (tu.to_reference(t) for t in (inp, weight, bias))

    ref_out = torch.ops.aten._nnpack_spatial_convolution(
        ref_inp, ref_weight, ref_bias, [1, 1], [1, 1]
    )
    res_out = flag_gems._nnpack_spatial_convolution(inp, weight, bias, [1, 1], [1, 1])

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.parametrize("kernel,padding", _KERNEL_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__nnpack_spatial_convolution_kernel_size(kernel, padding, value_range):
    inp, weight, bias = _conv_operands((2, 3, 19, 7), 6, value_range, kernel)
    ref_inp, ref_weight, ref_bias = (tu.to_reference(t) for t in (inp, weight, bias))

    ref_out = torch.ops.aten._nnpack_spatial_convolution(
        ref_inp, ref_weight, ref_bias, padding, [1, 1]
    )
    res_out = flag_gems._nnpack_spatial_convolution(inp, weight, bias, padding, [1, 1])

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.parametrize("padding,stride", _PAD_STRIDE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__nnpack_spatial_convolution_padding_stride(padding, stride, value_range):
    inp, weight, bias = _conv_operands((2, 3, 19, 17), 5, value_range)
    ref_inp, ref_weight, ref_bias = (tu.to_reference(t) for t in (inp, weight, bias))

    ref_out = torch.ops.aten._nnpack_spatial_convolution(
        ref_inp, ref_weight, ref_bias, padding, stride
    )
    res_out = flag_gems._nnpack_spatial_convolution(inp, weight, bias, padding, stride)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__nnpack_spatial_convolution_default_stride(value_range):
    # stride has the schema default [1, 1]; omitting it checks that the public
    # candidate signature implements that default instead of requiring it.
    inp, weight, bias = _conv_operands((2, 3, 19, 7), 6, value_range)
    ref_inp, ref_weight, ref_bias = (tu.to_reference(t) for t in (inp, weight, bias))

    ref_out = torch.ops.aten._nnpack_spatial_convolution(
        ref_inp, ref_weight, ref_bias, [1, 1]
    )
    res_out = flag_gems._nnpack_spatial_convolution(inp, weight, bias, [1, 1])

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__nnpack_spatial_convolution_bias(with_bias, value_range):
    inp, weight, bias = _conv_operands((2, 3, 19, 7), 6, value_range)
    bias = bias if with_bias else None
    ref_inp, ref_weight = tu.to_reference(inp), tu.to_reference(weight)
    ref_bias = tu.to_reference(bias)

    ref_out = torch.ops.aten._nnpack_spatial_convolution(
        ref_inp, ref_weight, ref_bias, [1, 1], [1, 1]
    )
    res_out = flag_gems._nnpack_spatial_convolution(inp, weight, bias, [1, 1], [1, 1])

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.parametrize("shape,out_channels", _SHAPE_CASES[:2])
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__nnpack_spatial_convolution_out(shape, out_channels, value_range):
    inp, weight, bias = _conv_operands(shape, out_channels, value_range)
    ref_inp, ref_weight, ref_bias = (tu.to_reference(t) for t in (inp, weight, bias))
    out_shape = torch.ops.aten._nnpack_spatial_convolution(
        ref_inp, ref_weight, ref_bias, [1, 1], [1, 1]
    ).shape

    # Both buffers start identical, so any region the kernel leaves unwritten
    # shows up as a value mismatch against the reference output.
    ref_out = torch.full(out_shape, 0.5, dtype=torch.float32, device=_CPU)
    res_out = torch.full(out_shape, 0.5, dtype=torch.float32, device=_CPU)
    torch.ops.aten._nnpack_spatial_convolution.out(
        ref_inp, ref_weight, ref_bias, [1, 1], [1, 1], out=ref_out
    )
    res_ret = flag_gems._nnpack_spatial_convolution(
        inp, weight, bias, [1, 1], [1, 1], out=res_out
    )

    assert res_ret is res_out
    tu.assert_result_close(res_ret, ref_out)


@pytest.mark.parametrize("operand", ["input", "weight"])
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test__nnpack_spatial_convolution_noncontiguous(operand, value_range):
    inp, weight, bias = _conv_operands((2, 3, 19, 7), 6, value_range)
    # Transposing and re-contiguating keeps the logical NCHW shape while the H/W
    # axes carry transposed strides, so the native kernel reads a genuinely
    # non-contiguous operand instead of a dense copy.
    if operand == "input":
        inp = inp.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    else:
        weight = weight.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    ref_inp, ref_weight, ref_bias = (tu.to_reference(t) for t in (inp, weight, bias))

    ref_out = torch.ops.aten._nnpack_spatial_convolution(
        ref_inp, ref_weight, ref_bias, [1, 1], [1, 1]
    )
    res_out = flag_gems._nnpack_spatial_convolution(inp, weight, bias, [1, 1], [1, 1])

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.parametrize("shape,out_channels,padding,stride", _BACKWARD_CASES)
def test__nnpack_spatial_convolution_backward(shape, out_channels, padding, stride):
    inp, weight, bias = _conv_operands(shape, out_channels, ["-1", "1"])
    inp.requires_grad_(True)
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    ref_inp = tu.to_reference(inp).requires_grad_(True)
    ref_weight = tu.to_reference(weight).requires_grad_(True)
    ref_bias = tu.to_reference(bias).requires_grad_(True)

    ref_out = torch.ops.aten._nnpack_spatial_convolution(
        ref_inp, ref_weight, ref_bias, padding, stride
    )
    res_out = flag_gems._nnpack_spatial_convolution(inp, weight, bias, padding, stride)

    tu.assert_result_close(res_out, ref_out)
    upstream = torch.randn_like(res_out)
    ref_grads = torch.autograd.grad(ref_out, (ref_inp, ref_weight, ref_bias), upstream)
    res_grads = torch.autograd.grad(res_out, (inp, weight, bias), upstream)

    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases([torch.float32]), quick=[]),
)
@pytest.mark.parametrize("kernel,padding,plane,payload_len", _SPECIAL_ROWS)
def test__nnpack_spatial_convolution_special_values(
    dtype, scenario, kernel, padding, plane, payload_len
):
    flat = tu.make_special_input(dtype, scenario).to(_CPU).flatten()
    tiles = -(-payload_len // flat.numel())
    inp = flat.repeat(tiles)[:payload_len].reshape(plane)
    weight = torch.ones(3, plane[1], kernel, kernel, dtype=dtype, device=_CPU)
    bias = torch.zeros(3, dtype=dtype, device=_CPU)
    ref_inp, ref_weight, ref_bias = (tu.to_reference(t) for t in (inp, weight, bias))

    ref_out = torch.ops.aten._nnpack_spatial_convolution(
        ref_inp, ref_weight, ref_bias, padding, [1, 1]
    )
    res_out = flag_gems._nnpack_spatial_convolution(inp, weight, bias, padding, [1, 1])

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test__nnpack_spatial_convolution_unsupported_dtype(dtype):
    inp, weight, bias = _conv_operands((2, 3, 19, 7), 6, ["-1", "1"])
    inp, weight, bias = inp.to(dtype), weight.to(dtype), bias.to(dtype)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._nnpack_spatial_convolution(inp, weight, bias, [1, 1], [1, 1])


@pytest.mark.parametrize("shape", _INVALID_INPUT_SHAPES)
def test__nnpack_spatial_convolution_invalid_input_shape(shape):
    inp = _cpu_input(shape, ["-1", "1"])
    weight = _cpu_input((4, 3, 3, 3), ["-1", "1"])
    bias = _cpu_input((4,), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._nnpack_spatial_convolution(inp, weight, bias, [1, 1], [1, 1])


@pytest.mark.parametrize("weight_shape", _BAD_WEIGHT_SHAPES)
def test__nnpack_spatial_convolution_invalid_weight_shape(weight_shape):
    inp = _cpu_input((2, 3, 19, 7), ["-1", "1"])
    weight = _cpu_input(weight_shape, ["-1", "1"])
    bias = _cpu_input((4,), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._nnpack_spatial_convolution(inp, weight, bias, [1, 1], [1, 1])


@pytest.mark.parametrize("padding", _BAD_PADDING)
def test__nnpack_spatial_convolution_negative_padding(padding):
    inp, weight, bias = _conv_operands((2, 3, 19, 7), 6, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._nnpack_spatial_convolution(inp, weight, bias, padding, [1, 1])


@pytest.mark.parametrize("stride", _BAD_STRIDE)
def test__nnpack_spatial_convolution_negative_stride(stride):
    inp, weight, bias = _conv_operands((2, 3, 19, 7), 6, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._nnpack_spatial_convolution(inp, weight, bias, [1, 1], stride)


def test__nnpack_spatial_convolution_non_tensor_weight():
    inp, _, bias = _conv_operands((2, 3, 19, 7), 6, ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems._nnpack_spatial_convolution(inp, [[[0.0]]], bias, [1, 1], [1, 1])
