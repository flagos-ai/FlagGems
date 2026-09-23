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

SILU_AND_MUL_WITH_CLAMP_LIMITS = [3.0, 7.0]


@pytest.mark.silu_and_mul_with_clamp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("limit", SILU_AND_MUL_WITH_CLAMP_LIMITS)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_silu_and_mul_with_clamp(shape, dtype, limit):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_gate = torch.clamp(ref_inp1, max=limit)
    ref_up = torch.clamp(ref_inp2, min=-limit, max=limit)
    ref_out = torch.mul(torch.nn.functional.silu(ref_gate), ref_up)
    with flag_gems.use_gems():
        res_out = flag_gems.silu_and_mul_with_clamp(inp1, inp2, limit)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)

    ref_inp1_grad, ref_inp2_grad = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )

    res_inp1_grad, res_inp2_grad = torch.autograd.grad(res_out, (inp1, inp2), out_grad)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_inp1_grad, ref_inp1_grad, dtype)
    utils.gems_assert_close(res_inp2_grad, ref_inp2_grad, dtype)


@pytest.mark.silu_and_mul_with_clamp_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("limit", SILU_AND_MUL_WITH_CLAMP_LIMITS)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_silu_and_mul_with_clamp_out(shape, dtype, limit):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_gate = torch.clamp(ref_inp1, max=limit)
    ref_up = torch.clamp(ref_inp2, min=-limit, max=limit)
    ref_out = torch.mul(torch.nn.functional.silu(ref_gate), ref_up)

    out = torch.empty_like(inp1)
    with flag_gems.use_gems():
        ret = flag_gems.silu_and_mul_with_clamp_out(inp1, inp2, out, limit)

    assert ret is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "layout", ["tail", "packed", "transpose", "broadcast", "empty"]
)
@pytest.mark.parametrize("limit", [0.0, 0.7, -0.7, 7.0])
def test_silu_clamp_scalar_boundaries_and_layouts(dtype, layout, limit):
    from flag_gems.fused.silu_and_mul_with_clamp import (
        silu_and_mul_with_clamp,
        silu_and_mul_with_clamp_out,
    )

    rounded = torch.tensor(limit, dtype=dtype).item()
    values = torch.tensor(
        [-float("inf"), -8, -rounded, 0, rounded, 8, float("inf"), float("nan")],
        device=flag_gems.device,
        dtype=dtype,
    )
    values = torch.cat(
        (
            values,
            torch.nextafter(values, torch.full_like(values, -float("inf"))),
            torch.nextafter(values, torch.full_like(values, float("inf"))),
        )
    )
    x = values.repeat(65)[:1539].reshape(3, 513)
    y = x.flip(-1).clone()
    if layout == "packed":
        packed = torch.cat((x, y), dim=-1)
        x, y = packed[:, :513], packed[:, 513:]
    elif layout == "transpose":
        x, y = x.T, y.T
    elif layout == "broadcast":
        y = y[:1]
    elif layout == "empty":
        x, y = x[:0], y[:0]
    x.requires_grad_()
    y.requires_grad_()
    # Spell out the original FP32 forward and backward, including its NaN
    # handling (minimum/maximum select the non-NaN operand).
    xf, yf = x.float(), y.float()
    threshold = torch.tensor(rounded, device=x.device, dtype=torch.float32)
    gate = torch.fmin(xf, threshold)
    up = torch.fmin(torch.fmax(yf, -threshold), threshold)
    ref = (gate / (1 + torch.exp(-gate))) * up
    result = silu_and_mul_with_clamp(x, y, limit)
    torch.testing.assert_close(result, ref.to(dtype), equal_nan=True)
    dout = torch.ones_like(result)
    dx, dy = torch.autograd.grad(result, (x, y), dout)
    sig = 1 / (1 + torch.exp(-gate))
    ref_dx = up * (sig * (1 + gate * (1 - sig))) * (xf <= rounded).float()
    ref_dy = (gate * sig) * ((yf >= -rounded) & (yf <= rounded)).float()
    torch.testing.assert_close(
        dx, ref_dx.sum_to_size(x.shape).to(dtype), equal_nan=True
    )
    torch.testing.assert_close(
        dy, ref_dy.sum_to_size(y.shape).to(dtype), equal_nan=True
    )
    # A strided output must bypass the linear-pointer path.
    storage = torch.full((*result.shape, 2), 42, dtype=dtype, device=x.device)
    out = storage[..., 0]
    assert silu_and_mul_with_clamp_out(x, y, out, limit) is out
    torch.testing.assert_close(out, result, equal_nan=True)
    assert torch.all(storage[..., 1] == 42)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon tuned path")
def test_silu_clamp_scalar_cache_miss_during_graph_capture():
    from flag_gems.fused.silu_and_mul_with_clamp import (
        _rounded_limit,
        silu_and_mul_with_clamp,
    )

    x = torch.randn((3, 513), device=flag_gems.device, dtype=torch.bfloat16)
    y = torch.randn_like(x)
    expected = silu_and_mul_with_clamp(x, y, 0.7)
    torch.cuda.synchronize()
    _rounded_limit.cache_clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = silu_and_mul_with_clamp(x, y, 0.7)
    graph.replay()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("requires_grad", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("mixed", [False, True])
def test_silu_clamp_broadcast_partial_gradients(requires_grad, mixed):
    from flag_gems.fused.silu_and_mul_with_clamp import silu_and_mul_with_clamp

    x = torch.randn((2, 1, 17), device=flag_gems.device, dtype=torch.float32)
    y = torch.randn(
        (1, 3, 1), device=x.device, dtype=torch.float16 if mixed else x.dtype
    )
    x.requires_grad_(requires_grad[0])
    y.requires_grad_(requires_grad[1])
    xr = x.detach().clone().requires_grad_(requires_grad[0])
    yr = y.detach().float().requires_grad_(requires_grad[1])
    limit = torch.tensor(0.7, dtype=x.dtype).item()
    gate = xr.clamp(max=limit)
    up = yr.clamp(min=-limit, max=limit)
    reference = torch.nn.functional.silu(gate) * up
    result = silu_and_mul_with_clamp(x, y, 0.7)
    # Expanded zero-stride gradient input exercises the strided backward path.
    grad = torch.randn((1, 3, 1), device=x.device).expand(result.shape)
    actual_inputs = [t for t in (x, y) if t.requires_grad]
    reference_inputs = [t for t in (xr, yr) if t.requires_grad]
    actual_grads = torch.autograd.grad(result, actual_inputs, grad)
    reference_grads = torch.autograd.grad(reference, reference_inputs, grad)
    torch.testing.assert_close(result, reference)
    for actual, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected.to(actual.dtype))


@pytest.mark.parametrize("partial", [False, True])
def test_silu_clamp_out_alias_during_tuning(partial):
    from flag_gems.fused.silu_and_mul_with_clamp import (
        silu_and_mul_with_clamp,
        silu_and_mul_with_clamp_out,
    )

    storage = torch.randn(2051, device=flag_gems.device)
    x = storage[:-1]
    y = torch.randn_like(x)
    reference_x = x.clone()
    out = storage[1:] if partial else x
    assert silu_and_mul_with_clamp_out(x, y, out, 0.7) is out
    expected = silu_and_mul_with_clamp(reference_x, y, 0.7)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["transpose", "channels_last", "different_strides"])
def test_silu_clamp_dense_layouts(layout):
    from flag_gems.fused.silu_and_mul_with_clamp import silu_and_mul_with_clamp

    x = torch.randn((2, 3, 7, 17), device=flag_gems.device)
    x = (
        x.to(memory_format=torch.channels_last)
        if layout == "channels_last"
        else x.transpose(1, 3)
    )
    y = torch.randn_like(x)
    if layout == "different_strides":
        y = y.contiguous()
    x.requires_grad_()
    y.requires_grad_()
    xr = x.detach().clone().requires_grad_()
    yr = y.detach().clone().requires_grad_()
    reference = torch.nn.functional.silu(xr.clamp(max=7.0)) * yr.clamp(-7.0, 7.0)
    result = silu_and_mul_with_clamp(x, y, 7.0)
    assert result.stride() == x.stride()
    grad = torch.randn_like(result)
    actual_grads = torch.autograd.grad(result, (x, y), grad)
    reference_grads = torch.autograd.grad(reference, (xr, yr), grad)
    torch.testing.assert_close(result, reference)
    for actual, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected)
