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

# Different ranks that broadcast against each other; every case has to reduce
# the upstream gradient back to the original operand shape.
BROADCAST_SHAPES = [
    ((3, 1, 5), (2, 3, 4, 5)),
    ((1, 5), (4, 5)),
    ((2, 3, 4, 5), (5,)),
    ((4, 1), (1, 3)),
    ((2, 3), (2, 3)),
]


def _reference_minimum_backward(x, y, out_grad):
    """Native autograd gradient on the reference device."""
    ref_x = utils.to_reference(x.detach(), True).requires_grad_()
    ref_y = utils.to_reference(y.detach(), True).requires_grad_()
    ref_grad = utils.to_reference(out_grad.detach(), True)
    ref_out = torch.minimum(ref_x, ref_y)
    return torch.autograd.grad(ref_out, (ref_x, ref_y), (ref_grad,))


@pytest.mark.minimum
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.minimum(ref_inp1, ref_inp2)
    res_out = flag_gems.minimum(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.minimum_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum_backward(shape, dtype):
    res_x = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    res_y = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_gx, ref_gy = _reference_minimum_backward(res_x, res_y, out_grad)
    res_out = flag_gems.minimum(res_x, res_y)
    res_gx, res_gy = torch.autograd.grad(res_out, (res_x, res_y), (out_grad,))

    utils.gems_assert_close(res_gx, ref_gx, dtype)
    utils.gems_assert_close(res_gy, ref_gy, dtype)


@pytest.mark.minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("needs_gx", [True, False])
def test_minimum_backward_single_side(dtype, needs_gx):
    shape = (37, 11)
    base_x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base_y = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_x = base_x.clone().requires_grad_(needs_gx)
    res_y = base_y.clone().requires_grad_(not needs_gx)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_gx, ref_gy = _reference_minimum_backward(base_x, base_y, out_grad)
    res_out = flag_gems.minimum(res_x, res_y)
    wanted = res_x if needs_gx else res_y
    (res_grad,) = torch.autograd.grad(res_out, (wanted,), (out_grad,))

    ref_grad = ref_gx if needs_gx else ref_gy
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    # The unused operand must not receive a gradient.
    assert (res_y.grad if needs_gx else res_x.grad) is None


@pytest.mark.minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum_backward_ties(dtype):
    """A tie splits the incoming gradient evenly between both operands."""
    shape = (64, 32)
    base_x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base_y = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Half of the elements are exact ties.
    base_y[::2] = base_x[::2]
    res_x = base_x.clone().requires_grad_(True)
    res_y = base_y.clone().requires_grad_(True)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_gx, ref_gy = _reference_minimum_backward(base_x, base_y, out_grad)
    res_out = flag_gems.minimum(res_x, res_y)
    res_gx, res_gy = torch.autograd.grad(res_out, (res_x, res_y), (out_grad,))

    utils.gems_assert_close(res_gx, ref_gx, dtype)
    utils.gems_assert_close(res_gy, ref_gy, dtype)
    # Sanity-check the tie rule itself, not only the parity with ATen.
    half = (utils.to_reference(out_grad, True) * 0.5).to(dtype)
    assert torch.equal(utils.to_reference(res_gx)[::2], half[::2])


@pytest.mark.minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum_backward_nan(dtype):
    """Both operands keep the incoming gradient where a comparison is NaN."""
    base_x = torch.tensor([float("nan"), 1.0, 3.0, float("nan"), 2.0], dtype=dtype)
    base_y = torch.tensor(
        [1.0, float("nan"), float("nan"), float("nan"), 0.0], dtype=dtype
    )
    res_x = base_x.to(flag_gems.device).requires_grad_(True)
    res_y = base_y.to(flag_gems.device).requires_grad_(True)
    out_grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype).to(flag_gems.device)

    ref_gx, ref_gy = _reference_minimum_backward(res_x, res_y, out_grad)
    res_out = flag_gems.minimum(res_x, res_y)
    res_gx, res_gy = torch.autograd.grad(res_out, (res_x, res_y), (out_grad,))

    utils.gems_assert_close(res_gx, ref_gx, dtype)
    utils.gems_assert_close(res_gy, ref_gy, dtype)
    # NaN elements are not a tie: the full gradient flows to both sides.
    ref_grad = utils.to_reference(out_grad)
    assert torch.equal(utils.to_reference(res_gx)[:4], ref_grad[:4])
    assert torch.equal(utils.to_reference(res_gy)[:4], ref_grad[:4])


@pytest.mark.minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum_backward_signed_zero(dtype):
    """+0.0 and -0.0 compare equal, so they split the gradient instead of losing."""
    base_x = torch.tensor([0.0, -0.0, 0.0, -0.0, 1.0], dtype=dtype)
    base_y = torch.tensor([-0.0, 0.0, 0.0, 0.0, 2.0], dtype=dtype)
    res_x = base_x.to(flag_gems.device).requires_grad_(True)
    res_y = base_y.to(flag_gems.device).requires_grad_(True)
    out_grad = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0], dtype=dtype).to(
        flag_gems.device
    )

    ref_gx, ref_gy = _reference_minimum_backward(res_x, res_y, out_grad)
    res_out = flag_gems.minimum(res_x, res_y)
    res_gx, res_gy = torch.autograd.grad(res_out, (res_x, res_y), (out_grad,))

    utils.gems_assert_close(res_gx, ref_gx, dtype)
    utils.gems_assert_close(res_gy, ref_gy, dtype)
    half = (utils.to_reference(out_grad, True) * 0.5).to(dtype)
    assert torch.equal(utils.to_reference(res_gx)[:4], half[:4])
    assert torch.equal(utils.to_reference(res_gy)[:4], half[:4])


@pytest.mark.minimum_backward
@pytest.mark.parametrize("shapes", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum_backward_broadcast(shapes, dtype):
    x_shape, y_shape = shapes
    base_x = torch.randn(x_shape, dtype=dtype, device=flag_gems.device)
    base_y = torch.randn(y_shape, dtype=dtype, device=flag_gems.device)
    res_x = base_x.clone().requires_grad_(True)
    res_y = base_y.clone().requires_grad_(True)
    out_grad = torch.randn(
        torch.broadcast_shapes(x_shape, y_shape),
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_gx, ref_gy = _reference_minimum_backward(base_x, base_y, out_grad)
    res_out = flag_gems.minimum(res_x, res_y)
    res_gx, res_gy = torch.autograd.grad(res_out, (res_x, res_y), (out_grad,))

    assert res_gx.shape == x_shape
    assert res_gy.shape == y_shape
    utils.gems_assert_close(res_gx, ref_gx, dtype)
    utils.gems_assert_close(res_gy, ref_gy, dtype)


@pytest.mark.minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum_backward_non_contiguous(dtype):
    """Transposed leaves have overlapping strides; gradients must still match."""
    base_x = torch.randn((7, 5, 3), dtype=dtype, device=flag_gems.device)
    base_y = torch.randn((7, 5, 3), dtype=dtype, device=flag_gems.device)
    x = base_x.permute(2, 0, 1)
    y = base_y.permute(2, 0, 1)
    assert not x.is_contiguous() and not y.is_contiguous()
    res_x = x.detach().requires_grad_(True)
    res_y = y.detach().requires_grad_(True)
    out_grad = torch.randn(x.shape, dtype=dtype, device=flag_gems.device)

    ref_gx, ref_gy = _reference_minimum_backward(x, y, out_grad)
    res_out = flag_gems.minimum(res_x, res_y)
    res_gx, res_gy = torch.autograd.grad(res_out, (res_x, res_y), (out_grad,))

    assert res_gx.shape == x.shape and res_gy.shape == y.shape
    utils.gems_assert_close(res_gx, ref_gx, dtype)
    utils.gems_assert_close(res_gy, ref_gy, dtype)


@pytest.mark.minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum_backward_direct_entry(dtype):
    """The direct FlagGems entry attaches its custom backward."""
    shape = (128, 96)
    base_x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base_y = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_gx, ref_gy = _reference_minimum_backward(base_x, base_y, out_grad)

    x = base_x.clone().requires_grad_(True)
    y = base_y.clone().requires_grad_(True)
    out = flag_gems.minimum(x, y)
    gx, gy = torch.autograd.grad(out, (x, y), (out_grad,))

    utils.gems_assert_close(gx, ref_gx, dtype)
    utils.gems_assert_close(gy, ref_gy, dtype)
    assert type(out.grad_fn).__name__ == "_MinimumBackward"


@pytest.mark.minimum_backward
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires two CUDA devices to verify the device guard",
)
@pytest.mark.parametrize("size", [1024, 65537])
def test_minimum_backward_device_guard(size):
    """Launch on the inputs' device even when another device is current."""
    with torch.cuda.device(1):
        x = torch.randn(size, device="cuda:1", requires_grad=True)
        y = torch.randn(size, device="cuda:1", requires_grad=True)
        grad = torch.randn_like(x)
        expected_x, expected_y = _reference_minimum_backward(x, y, grad)
        out = flag_gems.minimum(x, y)
    with torch.cuda.device(0):
        gx, gy = torch.autograd.grad(out, (x, y), grad)
        assert torch.cuda.current_device() == 0
    assert gx.device == x.device and gy.device == y.device
    utils.gems_assert_close(gx, expected_x, x.dtype)
    utils.gems_assert_close(gy, expected_y, y.dtype)


@pytest.mark.minimum_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_minimum_backward_zero_numel(dtype):
    shape = (0, 5)
    base_x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base_y = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_x = base_x.clone().requires_grad_(True)
    res_y = base_y.clone().requires_grad_(True)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.minimum(res_x, res_y)
    res_gx, res_gy = torch.autograd.grad(
        res_out, (res_x, res_y), (out_grad,), allow_unused=False
    )
    assert res_gx.shape == shape and res_gx.numel() == 0
    assert res_gy.shape == shape and res_gy.numel() == 0
