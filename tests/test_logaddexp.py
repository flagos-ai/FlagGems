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


@pytest.mark.logaddexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.logaddexp(ref_x, ref_y)

    res_out = flag_gems.logaddexp(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logaddexp_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.logaddexp.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.logaddexp_out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)


def _backward_reference(x, y, grad_output):
    """fp64 autograd reference for the same upstream gradient."""
    ref_x = utils.to_reference(x, True).detach().requires_grad_()
    ref_y = utils.to_reference(y, True).detach().requires_grad_()
    ref_grad = utils.to_reference(grad_output, True)
    ref_out = torch.logaddexp(ref_x, ref_y)
    return torch.autograd.grad(ref_out, (ref_x, ref_y), ref_grad)


@pytest.mark.logaddexp_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_backward(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_grad_x, ref_grad_y = _backward_reference(x, y, grad_output)

    out = flag_gems.logaddexp(x, y)
    grad_x, grad_y = torch.autograd.grad(out, (x, y), grad_output)

    assert grad_x.shape == x.shape and grad_y.shape == y.shape
    assert grad_x.dtype == dtype and grad_y.dtype == dtype
    utils.gems_assert_close(grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(grad_y, ref_grad_y, dtype)


@pytest.mark.logaddexp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_backward_broadcast(dtype):
    x = torch.randn((4, 1, 8), dtype=dtype, device=flag_gems.device, requires_grad=True)
    y = torch.randn((1, 5, 8), dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad_output = torch.ones((4, 5, 8), dtype=dtype, device=flag_gems.device)

    out = flag_gems.logaddexp(x, y)
    assert out.shape == (4, 5, 8)
    grad_x, grad_y = torch.autograd.grad(out, (x, y), grad_output)

    # Broadcast dimensions must be reduced back to each input's own shape.
    assert grad_x.shape == x.shape and grad_y.shape == y.shape
    assert grad_x.dtype == dtype and grad_y.dtype == dtype
    assert torch.isfinite(grad_x).all() and torch.isfinite(grad_y).all()

    if dtype == torch.float32:
        # Same accumulation width as the float64 reference: check the math.
        ref_grad_x, ref_grad_y = _backward_reference(x, y, grad_output)
        utils.gems_assert_close(grad_x, ref_grad_x, dtype)
        utils.gems_assert_close(grad_y, ref_grad_y, dtype)
    else:
        # A low-precision broadcast gradient is a sum of 4-5 separately rounded
        # terms, so any two correct implementations differ by a few dtype ulps;
        # hold the candidate to the native backward in the same dtype instead
        # of to the float64 reference (which the float32 branch already covers).
        native_x = x.detach().clone().requires_grad_()
        native_y = y.detach().clone().requires_grad_()
        native_out = torch.logaddexp(native_x, native_y)
        native_grad_x, native_grad_y = torch.autograd.grad(
            native_out, (native_x, native_y), grad_output
        )
        # Each element of a broadcast gradient adds up to five separately
        # rounded low-precision terms, so allow the usual one-ulp-per-term
        # budget that this suite applies to reduction outputs.
        utils.gems_assert_close(
            grad_x, utils.to_reference(native_grad_x), dtype, reduce_dim=5
        )
        utils.gems_assert_close(
            grad_y, utils.to_reference(native_grad_y), dtype, reduce_dim=4
        )


@pytest.mark.logaddexp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_backward_single_input(dtype):
    x = torch.randn((64, 32), dtype=dtype, device=flag_gems.device, requires_grad=True)
    y = torch.randn((64, 32), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn((64, 32), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True).detach().requires_grad_()
    ref_y = utils.to_reference(y, True).detach().requires_grad_()
    ref_grad = utils.to_reference(grad_output, True)
    ref_out = torch.logaddexp(ref_x, ref_y)
    (ref_grad_x,) = torch.autograd.grad(ref_out, (ref_x,), ref_grad)

    out = flag_gems.logaddexp(x, y)
    (grad_x,) = torch.autograd.grad(out, (x,), grad_output)

    utils.gems_assert_close(grad_x, ref_grad_x, dtype)


@pytest.mark.logaddexp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_backward_special_values(dtype):
    inf = float("inf")
    x_values = torch.tensor(
        [0.0, 1.0, -1.0, 1000.0, -1000.0, inf, inf, -inf, -inf, 3.5, 3.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    y_values = torch.tensor(
        [0.0, 1.0, 1.0, -1000.0, 1000.0, 1.0, inf, -1.0, -inf, 3.5, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    grad_values = torch.linspace(
        1.0, 2.0, x_values.numel(), dtype=dtype, device=flag_gems.device
    )

    x = x_values.clone().requires_grad_()
    y = y_values.clone().requires_grad_()

    ref_grad_x, ref_grad_y = _backward_reference(x, y, grad_values)

    out = flag_gems.logaddexp(x, y)
    grad_x, grad_y = torch.autograd.grad(out, (x, y), grad_values)

    # Equal inputs and same-sign infinities yield identical NaN patterns on both
    # paths; compare them as NaN rather than reporting a spurious mismatch.
    utils.gems_assert_close(grad_x, ref_grad_x, dtype, equal_nan=True)
    utils.gems_assert_close(grad_y, ref_grad_y, dtype, equal_nan=True)


@pytest.mark.logaddexp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_backward_entry_contract(dtype):
    """Only differentiable inputs build the wrapper node.

    The forward stays a plain pointwise kernel when no input can receive a
    gradient, so inference must not pay for an autograd.Function; when one
    input does require grad, the wrapper must own the backward. Both branches
    must produce the same values as the native forward.
    """
    x = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    y = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)

    ref_out = torch.logaddexp(utils.to_reference(x), utils.to_reference(y))

    out = flag_gems.logaddexp(x, y)
    assert out.requires_grad is False
    assert out.grad_fn is None
    utils.gems_assert_close(out, ref_out, dtype)

    grad_x = x.clone().requires_grad_()
    graph_out = flag_gems.logaddexp(grad_x, y)
    assert graph_out.requires_grad is True
    assert graph_out.grad_fn is not None
    utils.gems_assert_close(graph_out, ref_out, dtype)
    (dx,) = torch.autograd.grad(graph_out, (grad_x,), torch.ones_like(graph_out))
    assert dx.shape == x.shape and dx.dtype == dtype

    with torch.no_grad():
        no_grad_out = flag_gems.logaddexp(grad_x, y)
    assert no_grad_out.requires_grad is False
    assert no_grad_out.grad_fn is None


@pytest.mark.logaddexp_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logaddexp_backward_noncontiguous(dtype):
    """Transposed views must take the stride-aware path, not the flat kernel."""
    x = torch.randn((64, 8), dtype=dtype, device=flag_gems.device).t()
    y = torch.randn((64, 8), dtype=dtype, device=flag_gems.device).t()
    grad_output = torch.randn((8, 64), dtype=dtype, device=flag_gems.device)
    assert not x.is_contiguous() and not y.is_contiguous()

    x.requires_grad_()
    y.requires_grad_()
    ref_grad_x, ref_grad_y = _backward_reference(x, y, grad_output)

    out = flag_gems.logaddexp(x, y)
    grad_x, grad_y = torch.autograd.grad(out, (x, y), grad_output)

    assert grad_x.shape == x.shape and grad_y.shape == y.shape
    utils.gems_assert_close(grad_x, ref_grad_x, dtype)
    utils.gems_assert_close(grad_y, ref_grad_y, dtype)


@pytest.mark.logaddexp_backward
def test_logaddexp_backward_mixed_dtype():
    """A promoted pair keeps ATen's per-input gradient dtypes."""
    x = torch.randn((32, 16), dtype=torch.float16, device=flag_gems.device)
    y = torch.randn((32, 16), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn((32, 16), dtype=torch.float32, device=flag_gems.device)

    native_x = x.detach().clone().requires_grad_()
    native_y = y.detach().clone().requires_grad_()
    native_out = torch.logaddexp(native_x, native_y)
    assert native_out.dtype == torch.float32
    native_grad_x, native_grad_y = torch.autograd.grad(
        native_out, (native_x, native_y), grad_output
    )

    gems_x = x.detach().clone().requires_grad_()
    gems_y = y.detach().clone().requires_grad_()
    out = flag_gems.logaddexp(gems_x, gems_y)
    assert out.dtype == torch.float32
    grad_x, grad_y = torch.autograd.grad(out, (gems_x, gems_y), grad_output)

    assert grad_x.dtype == torch.float16 and grad_y.dtype == torch.float32
    utils.gems_assert_close(grad_x, utils.to_reference(native_grad_x), torch.float16)
    utils.gems_assert_close(grad_y, utils.to_reference(native_grad_y), torch.float32)


@pytest.mark.logaddexp_backward
def test_logaddexp_backward_autograd_broadcast():
    x = torch.randn((4, 1), device=flag_gems.device, requires_grad=True)
    y = torch.randn((1, 8), device=flag_gems.device, requires_grad=True)
    grad = torch.randn((4, 8), device=flag_gems.device)
    ref_x = utils.to_reference(x.detach()).requires_grad_(True)
    ref_y = utils.to_reference(y.detach()).requires_grad_(True)
    ref = torch.autograd.grad(
        torch.logaddexp(ref_x, ref_y), (ref_x, ref_y), utils.to_reference(grad)
    )
    out = flag_gems.logaddexp(x, y)
    assert type(out.grad_fn).__name__ == "LogaddexpFunctionBackward"
    result = torch.autograd.grad(out, (x, y), grad)
    for actual, expected in zip(result, ref):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        utils.gems_assert_close(actual, expected, expected.dtype)


@pytest.mark.logaddexp_backward
def test_logaddexp_backward_rejects_complex():
    x = torch.ones(
        (4,), dtype=torch.complex64, device=flag_gems.device, requires_grad=True
    )
    y = torch.ones(
        (4,), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    with pytest.raises(NotImplementedError, match="complex"):
        flag_gems.logaddexp(y, x)
