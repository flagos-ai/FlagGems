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

FWD_SHAPES = [
    (16, 128, 64),
    (1, 128, 64),
    (2, 100, 70),
    (5, 512, 129),
    (6, 512, 129),
    (4, 1024, 512),
    (2, 4096, 64),
    (0, 128, 64),
    (7, 256, 192),
    (48, 1024, 512),
    (192, 1024, 1024),
]


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch_size, in_features, out_features", FWD_SHAPES)
def test_linear_2d_with_bias(dtype, batch_size, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn(
        (batch_size, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    res_out = flag_gems.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch_size, in_features, out_features", FWD_SHAPES)
def test_linear_2d_without_bias(dtype, batch_size, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn(
        (batch_size, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight)
    res_out = flag_gems.linear(input_tensor, weight)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


FWD_3D_SHAPES = [
    (4, 8, 128, 64),
    (1, 1, 128, 64),
    (2, 2, 100, 70),
    (2, 4, 256, 192),
    (8, 24, 1024, 1024),
]


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("batch1, batch2, in_features, out_features", FWD_3D_SHAPES)
def test_linear_3d_with_bias(dtype, batch1, batch2, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn(
        (batch1, batch2, in_features), dtype=dtype, device=flag_gems.device
    )
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    res_out = flag_gems.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


FWD_1D_SHAPES = [
    (128, 64),
    (16, 16),
    (129, 70),
]


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("in_features, out_features", FWD_1D_SHAPES)
def test_linear_1d_with_bias(dtype, in_features, out_features):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2834: Skipping fp32 linear test on tsingmicro platform")

    input_tensor = torch.randn((in_features,), dtype=dtype, device=flag_gems.device)
    weight = torch.randn(
        (out_features, in_features), dtype=dtype, device=flag_gems.device
    )
    bias = torch.randn((out_features,), dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input_tensor, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_out = torch.nn.functional.linear(ref_input, ref_weight, ref_bias)
    res_out = flag_gems.linear(input_tensor, weight, bias)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=in_features)


@pytest.mark.linear
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("features", [64, 256, 513])
@pytest.mark.parametrize("batch", [1, 32])
def test_linear_bias_cancellation(dtype, features, batch):
    epsilon = {
        torch.float16: 2**-11,
        torch.bfloat16: 2**-8,
        torch.float32: 2**-22,
    }[dtype]
    cpu_input = torch.zeros((batch, features), dtype=dtype)
    cpu_input[:, :2] = 1
    cpu_weight = torch.zeros((33, features), dtype=dtype)
    cpu_weight[:, 0] = 1
    cpu_weight[::2, 1] = epsilon
    cpu_weight[1::2, 1] = -epsilon
    cpu_bias = torch.full((33,), -1, dtype=dtype)
    expected = torch.nn.functional.linear(
        cpu_input.float(), cpu_weight.float(), cpu_bias.float()
    ).to(dtype)
    inp = cpu_input.to(flag_gems.device)
    weight = cpu_weight.to(inp.device)
    bias = cpu_bias.to(inp.device)
    actual = flag_gems.linear(inp, weight, bias)
    utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.linear
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Non-vector bias is Ascend-specific"
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["matrix", "broadcast", "strided"])
def test_linear_nonvector_bias(dtype, layout):
    torch.manual_seed(16)
    cpu_input = (torch.randn(2, 3, 64) * 0.1).to(dtype)
    cpu_weight = (torch.randn(33, 64) * 0.1).to(dtype)
    cpu_bias = (torch.randn(2, 3, 33) * 0.1).to(dtype)
    if layout == "broadcast":
        cpu_bias = cpu_bias[:, :1, :]
    inp = cpu_input.to(flag_gems.device)
    weight = cpu_weight.to(inp.device)
    if layout == "strided":
        bias_storage = torch.empty(2, 3, 67, dtype=dtype, device=inp.device)
        bias = bias_storage[..., 1::2]
        bias.copy_(cpu_bias)
    else:
        bias = cpu_bias.to(inp.device)
    expected = torch.nn.functional.linear(
        cpu_input.float(), cpu_weight.float(), cpu_bias.float()
    ).to(dtype)
    actual = flag_gems.linear(inp, weight, bias)
    utils.gems_assert_close(actual.cpu(), expected, dtype, reduce_dim=64)
    utils.gems_assert_equal(bias.cpu(), cpu_bias)


@pytest.mark.linear
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Requires Ascend NPUGraph capture and replay",
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linear_nonvector_bias_graph(dtype):
    cpu_input = torch.ones((3, 64), dtype=dtype)
    cpu_weight = torch.zeros((33, 64), dtype=dtype)
    cpu_weight[:, :2] = 1
    inp = cpu_input.to(flag_gems.device)
    weight = cpu_weight.to(inp.device)
    bias = torch.ones((3, 33), dtype=dtype, device=inp.device)
    flag_gems.linear(inp, weight, bias)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = flag_gems.linear(inp, weight, bias)
    for value in (2, -3):
        inp.fill_(value)
        bias.fill_(value)
        actual.fill_(-123)
        graph.replay()
        torch.npu.synchronize()
        expected = torch.full((3, 33), 3 * value, dtype=dtype)
        utils.gems_assert_equal(actual.cpu(), expected)
    epsilon = {
        torch.float16: 2**-11,
        torch.bfloat16: 2**-8,
        torch.float32: 2**-22,
    }[dtype]
    cpu_input.zero_()
    cpu_input[:, :2] = 1
    cpu_weight.zero_()
    cpu_weight[:, 0] = 1
    cpu_weight[::2, 1] = epsilon
    cpu_weight[1::2, 1] = -epsilon
    inp.copy_(cpu_input)
    weight.copy_(cpu_weight)
    bias.fill_(-1)
    actual.fill_(-123)
    graph.replay()
    torch.npu.synchronize()
    expected = torch.nn.functional.linear(
        cpu_input.float(), cpu_weight.float(), torch.full((3, 33), -1.0)
    ).to(dtype)
    utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.linear
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Non-vector bias is Ascend-specific"
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["matrix", "broadcast", "strided"])
def test_linear_nonvector_bias_cancellation(dtype, layout):
    epsilon = {
        torch.float16: 2**-11,
        torch.bfloat16: 2**-8,
        torch.float32: 2**-22,
    }[dtype]
    shape = (3, 64) if layout == "matrix" else (2, 3, 64)
    bias_shape = (3, 33) if layout == "matrix" else (2, 1, 33)
    cpu_input = torch.zeros(shape, dtype=dtype)
    cpu_input[..., :2] = 1
    cpu_weight = torch.zeros((33, 64), dtype=dtype)
    cpu_weight[:, 0] = 1
    cpu_weight[::2, 1] = epsilon
    cpu_weight[1::2, 1] = -epsilon
    cpu_bias = torch.full(bias_shape, -1, dtype=dtype)
    expected = torch.nn.functional.linear(
        cpu_input.float(), cpu_weight.float(), cpu_bias.float()
    ).to(dtype)
    assert torch.count_nonzero(expected) == expected.numel()
    inp = cpu_input.to(flag_gems.device)
    weight = cpu_weight.to(inp.device)
    if layout == "strided":
        storage = torch.empty((2, 1, 67), dtype=dtype, device=inp.device)
        bias = storage[..., 1::2]
        bias.copy_(cpu_bias)
    else:
        bias = cpu_bias.to(inp.device)
    actual = flag_gems.linear(inp, weight, bias)
    utils.gems_assert_equal(actual.cpu(), expected)
    utils.gems_assert_equal(bias.cpu(), cpu_bias)
