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


@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("value_type", ["tensor", "scalar_tensor", "scalar"])
def test_masked_fill(shape, dtype, threshold, value_type):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    if value_type == "tensor":
        # Keep the value a 0-dim tensor (the .Tensor overload), but match the
        # input dtype: the on-device reference (native torch.masked_fill on XPU)
        # rejects a mismatched dtype ("data type not matched! kint64/kfloat16").
        value = torch.tensor(1024, dtype=dtype, device=flag_gems.device)
    elif value_type == "scalar_tensor":
        value = torch.scalar_tensor(1024, dtype=dtype, device=flag_gems.device)
    else:
        value = 1024

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    if torch.is_tensor(value):
        ref_out = torch.masked_fill(ref_inp, ref_mask, utils.to_reference(value))
    else:
        ref_out = torch.masked_fill(ref_inp, ref_mask, value)
    res_out = flag_gems.masked_fill(inp, mask, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_fill_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("value_type", ["tensor", "scalar_tensor", "scalar"])
def test_masked_fill_(shape, dtype, threshold, value_type):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    if value_type == "tensor":
        # Match the input dtype for the same reason as in ``test_masked_fill``:
        # the on-device reference rejects a mismatched value dtype.
        value = torch.tensor(1024, dtype=dtype, device=flag_gems.device)
    elif value_type == "scalar_tensor":
        value = torch.scalar_tensor(1024, dtype=dtype, device=flag_gems.device)
    else:
        value = 1024

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    if torch.is_tensor(value):
        ref_inp.masked_fill_(ref_mask, utils.to_reference(value))
    else:
        ref_inp.masked_fill_(ref_mask, value)
    flag_gems.masked_fill_(inp, mask, value)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.masked_fill_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_masked_fill_scalar(shape, dtype, threshold):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold
    value = 1024

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    ref_out = torch.masked_fill(ref_inp, ref_mask, value)
    res_out = flag_gems.masked_fill(inp, mask, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_fill_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("value", [1024, -512, 0.5, -0.25])
def test_masked_fill_scalar_(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    ref_inp.masked_fill_(ref_mask, value)
    flag_gems.masked_fill_(inp, mask, value)

    utils.gems_assert_equal(inp, ref_inp)


def _backward_mask(shape, dtype, mode, device, threshold=0.3):
    """Build the boolean mask used by the masked_fill backward tests."""
    if mode == "all_true":
        return torch.ones(shape, dtype=torch.bool, device=device)
    if mode == "all_false":
        return torch.zeros(shape, dtype=torch.bool, device=device)
    return torch.randn(shape, dtype=dtype, device=device) < threshold


@pytest.mark.masked_fill_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mask_mode", ["random", "all_true", "all_false"])
@pytest.mark.parametrize("value_type", ["scalar", "tensor"])
def test_masked_fill_backward_input(shape, dtype, mask_mode, value_type):
    """Gradient of ``input``: zero on masked positions, unchanged elsewhere."""
    device = flag_gems.device
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    mask = _backward_mask(shape, dtype, mask_mode, device)
    if value_type == "tensor":
        value = torch.tensor(1024, dtype=dtype, device=device)
    else:
        value = 1024

    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)
    ref_mask = utils.to_reference(mask)
    if torch.is_tensor(value):
        ref_value = utils.to_reference(value.detach().clone())
    else:
        ref_value = value

    res_out = flag_gems.masked_fill(inp, mask, value)
    ref_out = torch.masked_fill(ref_inp, ref_mask, ref_value)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad)

    (res_grad,) = torch.autograd.grad(res_out, (inp,), out_grad)
    (ref_grad,) = torch.autograd.grad(ref_out, (ref_inp,), ref_out_grad)

    utils.gems_assert_close(res_grad, ref_grad, dtype)


def _assert_value_grad_close(res, ref, dtype, grad_scale):
    """Compare the 0-dim gradient of a broadcast ``value``.

    The value gradient is a masked reduction, so the two sides accumulate the
    selected elements in float32 in a different order (ATen's tree reduction
    versus the two-stage Triton reduction) and cannot be bitwise equal. The
    tolerance therefore combines the dtype resolution with a rounding
    allowance that grows with ``sum(|grad_output|)``: ``256 * eps_float32`` of
    that magnitude covers the reduction ordering, while an actually wrong
    reduction (a missing mask, a doubled value, a wrong element count) still
    differs by a large multiple of it, or by more than the dtype resolution
    relative to ``|ref|``.
    """
    assert res.dtype == dtype
    res = res.detach().to("cpu", torch.float32)
    ref = ref.to(dtype).detach().to("cpu", torch.float32)
    assert res.shape == ref.shape == ()
    eps = torch.finfo(torch.float32).eps
    atol = 256.0 * eps * float(grad_scale)
    torch.testing.assert_close(res, ref, rtol=1e-5, atol=atol)


@pytest.mark.masked_fill_backward
@pytest.mark.filterwarnings(
    "ignore:Converting a tensor with requires_grad=True to a scalar"
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mask_mode", ["random", "all_true", "all_false"])
@pytest.mark.parametrize("grad_inputs", ["both", "value_only"])
def test_masked_fill_backward_tensor_value(shape, dtype, mask_mode, grad_inputs):
    """Differentiable 0-dim ``value``: masked gradients are summed."""
    device = flag_gems.device
    inp = torch.randn(shape, dtype=dtype, device=device)
    if grad_inputs == "both":
        inp.requires_grad_(True)
    mask = _backward_mask(shape, dtype, mask_mode, device)
    value = torch.tensor(1024, dtype=dtype, device=device, requires_grad=True)

    ref_inp = utils.to_reference(inp.detach().clone())
    if grad_inputs == "both":
        ref_inp.requires_grad_(True)
    ref_mask = utils.to_reference(mask)
    ref_value = utils.to_reference(value.detach().clone()).requires_grad_(True)

    res_out = flag_gems.masked_fill(inp, mask, value)
    ref_out = torch.masked_fill(ref_inp, ref_mask, ref_value)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad)

    if grad_inputs == "both":
        res_in_grad, res_value_grad = torch.autograd.grad(
            res_out, (inp, value), out_grad
        )
        ref_in_grad, ref_value_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_value), ref_out_grad
        )
        utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
    else:
        (res_value_grad,) = torch.autograd.grad(res_out, (value,), out_grad)
        (ref_value_grad,) = torch.autograd.grad(ref_out, (ref_value,), ref_out_grad)

    if mask_mode == "all_false":
        # Nothing is selected, so the broadcast value receives exactly zero.
        assert res_value_grad.item() == 0.0
    _assert_value_grad_close(
        res_value_grad, ref_value_grad, dtype, out_grad.abs().double().sum()
    )


@pytest.mark.masked_fill_backward
@pytest.mark.filterwarnings(
    "ignore:Converting a tensor with requires_grad=True to a scalar"
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("mask_shape", [(64, 1), (1, 512), (512,)])
def test_masked_fill_backward_broadcast_mask(dtype, mask_shape):
    """A mask broadcast over the input keeps the same masked-sum semantics."""
    device = flag_gems.device
    shape = (64, 512)
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    mask = torch.randn(mask_shape, dtype=dtype, device=device) < 0.3
    value = torch.tensor(-512, dtype=dtype, device=device, requires_grad=True)

    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)
    ref_mask = utils.to_reference(mask)
    ref_value = utils.to_reference(value.detach().clone()).requires_grad_(True)

    res_out = flag_gems.masked_fill(inp, mask, value)
    ref_out = torch.masked_fill(ref_inp, ref_mask, ref_value)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad)

    res_in_grad, res_value_grad = torch.autograd.grad(res_out, (inp, value), out_grad)
    ref_in_grad, ref_value_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_value), ref_out_grad
    )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
    _assert_value_grad_close(
        res_value_grad, ref_value_grad, dtype, out_grad.abs().double().sum()
    )


@pytest.mark.masked_fill_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_masked_fill_backward_non_contiguous(dtype):
    """Non-contiguous inputs must produce the same gradients."""
    device = flag_gems.device
    base = torch.randn((128, 64), dtype=dtype, device=device)
    inp = base.t()
    assert not inp.is_contiguous()
    inp.requires_grad_(True)
    mask = torch.randn(inp.shape, dtype=dtype, device=device) < 0.3
    value = 1024

    ref_base = utils.to_reference(base.detach().clone())
    ref_inp = ref_base.t().requires_grad_(True)
    assert not ref_inp.is_contiguous()
    ref_mask = utils.to_reference(mask)

    res_out = flag_gems.masked_fill(inp, mask, value)
    ref_out = torch.masked_fill(ref_inp, ref_mask, value)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad)

    (res_grad,) = torch.autograd.grad(res_out, (inp,), out_grad)
    (ref_grad,) = torch.autograd.grad(ref_out, (ref_inp,), ref_out_grad)

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.masked_fill_backward
@pytest.mark.parametrize("selected", [False, True])
def test_masked_fill_backward_device_scalar(selected):
    inp = torch.tensor(2.0, device=flag_gems.device, requires_grad=True)
    value = torch.tensor(3.0, device=flag_gems.device, requires_grad=True)
    mask = torch.tensor(selected, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_(True)
    ref_value = utils.to_reference(value.detach()).requires_grad_(True)
    ref = torch.masked_fill(ref_inp, utils.to_reference(mask), ref_value)
    actual = flag_gems.masked_fill(inp, mask, value)
    utils.gems_assert_equal(actual, ref)
    expected = torch.autograd.grad(ref, (ref_inp, ref_value))
    result = torch.autograd.grad(actual, (inp, value))
    for grad, ref_grad in zip(result, expected):
        utils.gems_assert_equal(grad, ref_grad)
