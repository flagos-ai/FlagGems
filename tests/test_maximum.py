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


@pytest.mark.maximum
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.maximum(ref_inp1, ref_inp2)
    res_out = flag_gems.maximum(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.maximum
@pytest.mark.parametrize("cuda_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cpu_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cpu_operand", ["self", "other"])
def test_maximum_cpu_scalar_operand(cuda_dtype, cpu_dtype, cpu_operand):
    x = torch.linspace(
        0.25, 4.0, 64, device=flag_gems.device, dtype=cuda_dtype
    ).reshape(8, 8)
    scalar = torch.tensor(1.5, device="cpu", dtype=cpu_dtype)
    inp1, inp2 = (scalar, x) if cpu_operand == "self" else (x, scalar)

    ref_out = torch.maximum(utils.to_reference(inp1), utils.to_reference(inp2))
    res_out = flag_gems.maximum(inp1, inp2)

    assert res_out.device == x.device
    assert res_out.dtype == ref_out.dtype == cuda_dtype
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.maximum
@pytest.mark.skipif(
    not utils.fp64_is_supported, reason="float64 is not supported on this device"
)
@pytest.mark.parametrize(
    "value, scalar_dtype",
    [
        (1.0 + 2.0**-52, torch.float64),
        (2**24 + 1, torch.int64),
        (2**53 + 2**29, torch.int64),
    ],
    ids=["fp64", "int64-2^24", "int64-2^53"],
)
@pytest.mark.parametrize("cpu_operand", ["self", "other"])
def test_maximum_cpu_scalar_precision(value, scalar_dtype, cpu_operand):
    inp = torch.ones((2, 3), dtype=torch.float64, device=flag_gems.device)
    # None of these values survives a trip through fp32.
    scalar = torch.tensor(value, dtype=scalar_dtype)
    inp1, inp2 = (scalar, inp) if cpu_operand == "self" else (inp, scalar)

    ref_out = torch.maximum(utils.to_reference(inp1), utils.to_reference(inp2))
    res_out = flag_gems.maximum(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.maximum
@pytest.mark.parametrize("cpu_operand", ["self", "other"])
def test_maximum_rejects_non_scalar_cpu_operand(cpu_operand):
    inp = torch.randn((2, 19, 7), device=flag_gems.device)
    cpu_tensor = torch.randn(7)
    inp1, inp2 = (cpu_tensor, inp) if cpu_operand == "self" else (inp, cpu_tensor)

    with pytest.raises(RuntimeError, match="same device"):
        torch.maximum(inp1, inp2)
    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.maximum(inp1, inp2)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def _maximum_backward_grads(x, y, grad, equal_nan=False):
    """Run the FlagGems forward/backward pair and its native reference."""
    ref_x = utils.to_reference(x.detach()).requires_grad_(x.requires_grad)
    ref_y = utils.to_reference(y.detach()).requires_grad_(y.requires_grad)
    ref_grad = utils.to_reference(grad)

    res_out = flag_gems.maximum(x, y)
    ref_out = torch.maximum(ref_x, ref_y)
    utils.gems_assert_close(res_out, ref_out, ref_out.dtype, equal_nan=equal_nan)

    def operands(out, inputs, upstream):
        wanted = tuple(item for item in inputs if item.requires_grad)
        grads = iter(torch.autograd.grad(out, wanted, (upstream,), allow_unused=True))
        return tuple(next(grads) if item.requires_grad else None for item in inputs)

    return (
        res_out,
        operands(res_out, (x, y), grad),
        operands(ref_out, (ref_x, ref_y), ref_grad),
    )


def _assert_matching_grads(res_grads, ref_grads, equal_nan=False):
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        if ref_grad is None:
            assert res_grad is None
            continue
        assert res_grad is not None
        assert res_grad.shape == ref_grad.shape
        assert res_grad.dtype == ref_grad.dtype
        utils.gems_assert_close(res_grad, ref_grad, ref_grad.dtype, equal_nan=equal_nan)


@pytest.mark.maximum_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum_backward(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    res_out, res_grads, ref_grads = _maximum_backward_grads(x, y, grad)

    # The candidate must really run its own autograd node.
    assert res_out.grad_fn is not None
    assert res_out.grad_fn.__class__.__name__ == "MaximumFunctionBackward"
    _assert_matching_grads(res_grads, ref_grads)


@pytest.mark.maximum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum_backward_ties(dtype):
    base = torch.arange(64, dtype=torch.float32, device=flag_gems.device)
    x = (base % 5).to(dtype)
    y = ((base * 3) % 5).to(dtype)
    y[7] = x[7]  # an exact tie that survives the dtype conversion
    x.requires_grad_()
    y.requires_grad_()
    grad = torch.randn(64, dtype=dtype, device=flag_gems.device)

    _, res_grads, ref_grads = _maximum_backward_grads(x, y, grad)
    _assert_matching_grads(res_grads, ref_grads)


@pytest.mark.maximum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum_backward_special_values(dtype):
    values = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.0, -1.0]
    x = torch.tensor(
        values + [1.0, 2.0, 3.0], dtype=dtype, device=flag_gems.device
    ).requires_grad_()
    y = torch.tensor(
        [1.0, 2.0, 3.0] + values, dtype=dtype, device=flag_gems.device
    ).requires_grad_()
    grad = torch.arange(1.0, 11.0, dtype=dtype, device=flag_gems.device)

    _, res_grads, ref_grads = _maximum_backward_grads(x, y, grad, equal_nan=True)
    _assert_matching_grads(res_grads, ref_grads, equal_nan=True)


@pytest.mark.maximum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum_backward_signed_zeros(dtype):
    x = torch.tensor(
        [0.0, -0.0, 0.0, -0.0, 1.0, -1.0], dtype=dtype, device=flag_gems.device
    ).requires_grad_()
    y = torch.tensor(
        [-0.0, 0.0, 0.0, -0.0, -1.0, 1.0], dtype=dtype, device=flag_gems.device
    ).requires_grad_()
    grad = torch.ones(6, dtype=dtype, device=flag_gems.device)

    _, res_grads, ref_grads = _maximum_backward_grads(x, y, grad)
    _assert_matching_grads(res_grads, ref_grads)


@pytest.mark.maximum_backward
@pytest.mark.parametrize("side", ["self", "other"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum_backward_single_sided(side, dtype):
    requires_grad = [side == "self", side == "other"]
    x = torch.randn(37, dtype=dtype, device=flag_gems.device).requires_grad_(
        requires_grad[0]
    )
    y = torch.randn(37, dtype=dtype, device=flag_gems.device).requires_grad_(
        requires_grad[1]
    )
    grad = torch.randn(37, dtype=dtype, device=flag_gems.device)

    _, res_grads, ref_grads = _maximum_backward_grads(x, y, grad)
    _assert_matching_grads(res_grads, ref_grads)
    skipped = 1 if side == "self" else 0
    assert res_grads[skipped] is None and ref_grads[skipped] is None


@pytest.mark.maximum_backward
@pytest.mark.parametrize(
    "shape, other_shape",
    [((4, 3), (3,)), ((2, 1, 4), (1, 3, 1)), ((5, 1), (1, 7)), ((6,), (1,))],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum_backward_broadcast(shape, other_shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    y = torch.randn(other_shape, dtype=dtype, device=flag_gems.device).requires_grad_()
    grad = torch.randn(
        torch.broadcast_shapes(shape, other_shape),
        dtype=dtype,
        device=flag_gems.device,
    )

    _, res_grads, ref_grads = _maximum_backward_grads(x, y, grad)
    _assert_matching_grads(res_grads, ref_grads)
    assert res_grads[0].shape == shape
    assert res_grads[1].shape == other_shape


@pytest.mark.maximum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum_backward_expanded_grad(dtype):
    x = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    y = torch.randn((7, 13), dtype=dtype, device=flag_gems.device).requires_grad_()
    grad = torch.randn((7, 1), dtype=dtype, device=flag_gems.device).expand(7, 13)

    _, res_grads, ref_grads = _maximum_backward_grads(x, y, grad)
    _assert_matching_grads(res_grads, ref_grads)


@pytest.mark.maximum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum_backward_non_contiguous(dtype):
    x = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    y = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    x.requires_grad_()
    y.requires_grad_()
    grad = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()

    assert not x.is_contiguous()
    _, res_grads, ref_grads = _maximum_backward_grads(x, y, grad)
    _assert_matching_grads(res_grads, ref_grads)


@pytest.mark.maximum_backward
@pytest.mark.parametrize(
    "self_dtype, other_dtype",
    [(torch.float32, torch.bfloat16), (torch.float16, torch.float32)],
)
def test_maximum_backward_mixed_dtype(self_dtype, other_dtype):
    x = torch.randn((33, 17), dtype=self_dtype, device=flag_gems.device)
    y = torch.randn((33, 17), dtype=other_dtype, device=flag_gems.device)
    x.requires_grad_()
    y.requires_grad_()
    grad = torch.randn((33, 17), dtype=torch.float32, device=flag_gems.device)

    _, res_grads, ref_grads = _maximum_backward_grads(x, y, grad)
    _assert_matching_grads(res_grads, ref_grads)
    assert res_grads[0].dtype == self_dtype
    assert res_grads[1].dtype == other_dtype


@pytest.mark.maximum_backward
@pytest.mark.skipif(
    not utils.fp64_is_supported, reason="float64 is not supported on this device"
)
def test_maximum_backward_float64():
    # float64 is part of the benchmark dtype set when the device supports it.
    x = torch.randn((37, 11), dtype=torch.float64, device=flag_gems.device)
    y = torch.randn((37, 11), dtype=torch.float64, device=flag_gems.device)
    x.requires_grad_()
    y.requires_grad_()
    grad = torch.randn((37, 11), dtype=torch.float64, device=flag_gems.device)

    _, res_grads, ref_grads = _maximum_backward_grads(x, y, grad)
    _assert_matching_grads(res_grads, ref_grads)


@pytest.mark.maximum_backward
def test_maximum_backward_empty_broadcast():
    x = torch.empty((0, 4), device=flag_gems.device, requires_grad=True)
    y = torch.ones((1, 4), device=flag_gems.device, requires_grad=True)
    grad = torch.empty_like(x)
    ref_x = utils.to_reference(x.detach()).requires_grad_(True)
    ref_y = utils.to_reference(y.detach()).requires_grad_(True)
    expected = torch.autograd.grad(
        torch.maximum(ref_x, ref_y), (ref_x, ref_y), utils.to_reference(grad)
    )
    result = torch.autograd.grad(flag_gems.maximum(x, y), (x, y), grad)
    for actual, ref in zip(result, expected):
        utils.gems_assert_equal(actual, ref)
