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

# Vendor backends ship their own forward-only ``log2`` (for example
# ``runtime/backend/_kunlunxin/ops/log2.py``), which replaces the generic
# implementation that carries the autograd bridge.  Those backends keep the
# native ATen backward, so the backward cases below do not apply to them.
_BACKWARD_BRIDGE = "runtime.backend" not in flag_gems.log2.__module__
_needs_bridge = pytest.mark.skipif(
    not _BACKWARD_BRIDGE,
    reason="this backend's log2 has no FlagGems autograd bridge",
)

# ``d/dx log2(x) = g / (x * ln 2)`` is defined for every real ``x`` except the
# way the two divisions round; ATen still returns a finite gradient for
# negative inputs even though the forward value is NaN, and both zeros and the
# infinities have to keep their sign.  The table below pins those cases
# against native autograd.
_SPECIAL_X = [
    0.0,
    -0.0,
    -1.0,
    -2.0,
    -0.5,
    -1e-30,
    1e-30,
    1.0,
    2.0,
    0.5,
    1e30,
    -1e30,
    float("inf"),
    float("-inf"),
    float("nan"),
]
_SPECIAL_GRAD = [1.0, -1.0, 0.0, 2.5, float("inf"), float("nan")]


@pytest.mark.log2
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log2(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log2(ref_inp)
    res_out = flag_gems.log2(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log2
@_needs_bridge
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log2_forward_requires_grad(shape, dtype):
    """The autograd bridge must not change the forward values."""
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.25

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log2(ref_inp)
    res_out = flag_gems.log2(inp.clone().requires_grad_(True))

    assert res_out.requires_grad
    assert res_out.grad_fn is not None
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log2_backward
@_needs_bridge
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log2_backward(shape, dtype):
    """Compare the input gradient of ``log2`` through the full autograd graph."""
    # Positive inputs keep the forward finite; the negative/zero/NaN domain is
    # covered by the special-value case below.
    res_inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.25
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(res_inp, True).requires_grad_(True)
    ref_grad = utils.to_reference(out_grad, True)
    ref_out = torch.log2(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, ref_grad)[0]

    res_inp = res_inp.clone().requires_grad_(True)
    res_out = flag_gems.log2(res_inp)
    assert "_Log2Function" in type(res_out.grad_fn).__name__
    res_in_grad = torch.autograd.grad(res_out, res_inp, out_grad)[0]

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.log2_backward
@_needs_bridge
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log2_backward_special_values(dtype):
    """Signed zeros, negative inputs, extremes, NaN and Inf in both arguments."""
    x = torch.tensor(_SPECIAL_X, dtype=dtype, device=flag_gems.device)
    grad = torch.tensor(_SPECIAL_GRAD, dtype=dtype, device=flag_gems.device)
    # Full cartesian product so every upstream gradient meets every input.
    x = x.repeat_interleave(grad.numel())
    grad = grad.repeat(x.numel() // grad.numel())

    # This case pins the in-dtype rounding, not a wider-type value, so the
    # reference is native autograd in the same dtype on the same device.
    ref_inp = x.clone().requires_grad_(True)
    ref_in_grad = torch.autograd.grad(torch.log2(ref_inp), ref_inp, grad)[0]

    res_inp = x.clone().requires_grad_(True)
    res_out = flag_gems.log2(res_inp)
    res_in_grad = torch.autograd.grad(res_out, res_inp, grad)[0]

    # Same dtype and same non-finite pattern on both sides.
    assert res_in_grad.dtype == ref_in_grad.dtype
    assert torch.equal(torch.isnan(res_in_grad), torch.isnan(ref_in_grad))
    assert torch.equal(torch.isinf(res_in_grad), torch.isinf(ref_in_grad))
    assert torch.equal(
        torch.sign(res_in_grad[torch.isinf(res_in_grad)]),
        torch.sign(ref_in_grad[torch.isinf(ref_in_grad)]),
    )
    finite = torch.isfinite(ref_in_grad)
    torch.testing.assert_close(
        res_in_grad[finite], ref_in_grad[finite], rtol=0.0, atol=0.0
    )


@pytest.mark.log2_backward
@_needs_bridge
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log2_backward_non_contiguous(dtype):
    """Strided input and a stride-0 (broadcast) upstream gradient."""
    base = torch.rand(48, 64, dtype=dtype, device=flag_gems.device) + 0.25
    x = base.t()
    assert not x.is_contiguous()

    out_grad = torch.randn(1, 48, dtype=dtype, device=flag_gems.device).expand(64, 48)
    assert out_grad.stride() == (0, 1)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(out_grad, True)
    ref_out = torch.log2(ref_inp)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, ref_grad)[0]

    res_inp = x.clone().requires_grad_(True)
    res_out = flag_gems.log2(res_inp)
    res_in_grad = torch.autograd.grad(res_out, res_inp, out_grad)[0]

    assert res_in_grad.shape == ref_in_grad.shape
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.log2_backward
@_needs_bridge
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log2_backward_permuted_3d(dtype):
    """A three-axis permutation keeps three independent strides."""
    base = torch.rand(8, 12, 16, dtype=dtype, device=flag_gems.device) + 0.25
    x = base.permute(2, 0, 1)
    out_grad = torch.randn(16, 8, 12, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(x, True).requires_grad_(True)
    ref_grad = utils.to_reference(out_grad, True)
    ref_in_grad = torch.autograd.grad(torch.log2(ref_inp), ref_inp, ref_grad)[0]

    res_inp = x.clone().requires_grad_(True)
    res_out = flag_gems.log2(res_inp)
    res_in_grad = torch.autograd.grad(res_out, res_inp, out_grad)[0]

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.log2_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log2_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = ref_inp.log2_()
    res_out = flag_gems.log2_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
