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

# (batch, n1, dim) shapes covering small/medium/large cases; n2 is derived per test
CDIST_FB_SHAPES = [(2, 16, 32), (4, 32, 64), (8, 64, 128)]

# p-norm orders exercising the L1/L2/L-inf special cases and the general path
CDIST_FB_P = [1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]


def _assert_cdist_fb_close(res_out, ref_out, dtype, p):
    # The p-norm backward for p < 1 has an extremely large dynamic range in
    # fp32 (|diff|^(p-1) amplifies small differences), so even ATen CUDA vs
    # CPU disagree at the ~1e-2 relative level. Use a relaxed tolerance there.
    if p < 1.0:
        res_out = res_out.to(device=ref_out.device)
        torch.testing.assert_close(res_out, ref_out, atol=1e-2, rtol=1e-2)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cdist_forward_backward
@pytest.mark.parametrize("shape", CDIST_FB_SHAPES)
# _cdist_backward uses intermediate fp32 accumulation; only float32 is numerically stable
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("p", [0.5] + CDIST_FB_P)
def test_cdist_forward_backward(shape, dtype, p):
    batch, n1, dim = shape
    n2 = n1 // 2 + 1  # Use different n2 for variety
    shape1 = (batch, n1, dim)
    shape2 = (batch, n2, dim)

    res_x1 = torch.randn(shape1, dtype=dtype, device=flag_gems.device)
    res_x2 = torch.randn(shape2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)

    ref_cdist = torch.cdist(ref_x1, ref_x2, p=p)
    res_cdist = ref_cdist.clone().to(flag_gems.device)
    ref_grad = torch.randn(ref_cdist.shape, dtype=dtype, device=ref_cdist.device)
    res_grad = ref_grad.clone().to(flag_gems.device)

    ref_gx1 = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, p, ref_cdist)
    ref_gx2 = torch.ops.aten._cdist_backward(
        ref_grad.transpose(-2, -1), ref_x2, ref_x1, p, ref_cdist.transpose(-2, -1)
    )
    res_gx1, res_gx2 = flag_gems.cdist_forward_backward(
        res_grad, res_x1, res_x2, p, res_cdist
    )

    assert res_gx1.shape == res_x1.shape
    assert res_gx2.shape == res_x2.shape
    _assert_cdist_fb_close(res_gx1, ref_gx1, dtype, p)
    _assert_cdist_fb_close(res_gx2, ref_gx2, dtype, p)


@pytest.mark.cdist_forward_backward
@pytest.mark.parametrize("p", CDIST_FB_P)
def test_cdist_forward_backward_zero_distance(p, dtype=torch.float32):
    """Guard against dist == 0 / diff == 0 lanes (identical points)."""
    batch, n1, n2, dim = 2, 5, 6, 8
    res_x1 = torch.randn(batch, n1, dim, dtype=dtype, device=flag_gems.device)
    res_x2 = torch.randn(batch, n2, dim, dtype=dtype, device=flag_gems.device)
    # Force some identical rows so that cdist contains exact zeros and the
    # corresponding diff vectors are exactly zero (dist==0 / diff==0 guards).
    res_x2[:, 0] = res_x1[:, 0]
    res_x2[:, 1] = res_x1[:, 2]

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_cdist = torch.cdist(ref_x1, ref_x2, p=p)
    res_cdist = ref_cdist.clone().to(flag_gems.device)
    ref_grad = torch.randn(ref_cdist.shape, dtype=dtype, device=ref_cdist.device)
    res_grad = ref_grad.clone().to(flag_gems.device)

    ref_gx1 = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, p, ref_cdist)
    ref_gx2 = torch.ops.aten._cdist_backward(
        ref_grad.transpose(-2, -1), ref_x2, ref_x1, p, ref_cdist.transpose(-2, -1)
    )
    res_gx1, res_gx2 = flag_gems.cdist_forward_backward(
        res_grad, res_x1, res_x2, p, res_cdist
    )

    _assert_cdist_fb_close(res_gx1, ref_gx1, dtype, p)
    _assert_cdist_fb_close(res_gx2, ref_gx2, dtype, p)


@pytest.mark.cdist_forward_backward
def test_cdist_forward_backward_padded_lanes_no_nan(dtype=torch.float32):
    """Padded reduction lanes must not fold inf * 0 == NaN into the gradient.

    With BLOCK 64 and n2=1 the masked lanes compute |1e10|^(p-1) * grad, i.e.
    inf * 0; before masking that NaN leaked into the accumulated gradient.
    """
    batch, n1, n2, dim = 1, 1, 1, 1
    big = 1e10
    res_x1 = torch.full((batch, n1, dim), big, dtype=dtype, device=flag_gems.device)
    res_x2 = torch.full((batch, n2, dim), big, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_cdist = torch.cdist(ref_x1, ref_x2, p=5.0)
    res_cdist = ref_cdist.clone().to(flag_gems.device)
    ref_grad = torch.ones(ref_cdist.shape, dtype=dtype, device=ref_cdist.device)
    res_grad = ref_grad.clone().to(flag_gems.device)

    ref_gx1 = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, 5.0, ref_cdist)
    ref_gx2 = torch.ops.aten._cdist_backward(
        ref_grad.transpose(-2, -1), ref_x2, ref_x1, 5.0, ref_cdist.transpose(-2, -1)
    )
    res_gx1, res_gx2 = flag_gems.cdist_forward_backward(
        res_grad, res_x1, res_x2, 5.0, res_cdist
    )

    assert torch.isfinite(res_gx1).all(), f"non-finite grad_x1: {res_gx1}"
    assert torch.isfinite(res_gx2).all(), f"non-finite grad_x2: {res_gx2}"
    utils.gems_assert_close(res_gx1, ref_gx1, dtype)
    utils.gems_assert_close(res_gx2, ref_gx2, dtype)


@pytest.mark.cdist_forward_backward
def test_cdist_forward_backward_p_zero(dtype=torch.float32):
    """p == 0 (zero-norm distance) has an identically zero gradient."""
    batch, n1, n2, dim = 2, 4, 3, 5
    res_x1 = torch.randn(batch, n1, dim, dtype=dtype, device=flag_gems.device)
    res_x2 = torch.randn(batch, n2, dim, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_cdist = torch.cdist(ref_x1, ref_x2, p=0.0)
    res_cdist = ref_cdist.clone().to(flag_gems.device)
    ref_grad = torch.randn(ref_cdist.shape, dtype=dtype, device=ref_cdist.device)
    res_grad = ref_grad.clone().to(flag_gems.device)

    ref_gx1 = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, 0.0, ref_cdist)
    ref_gx2 = torch.ops.aten._cdist_backward(
        ref_grad.transpose(-2, -1), ref_x2, ref_x1, 0.0, ref_cdist.transpose(-2, -1)
    )
    res_gx1, res_gx2 = flag_gems.cdist_forward_backward(
        res_grad, res_x1, res_x2, 0.0, res_cdist
    )

    utils.gems_assert_close(res_gx1, ref_gx1, dtype)
    utils.gems_assert_close(res_gx2, ref_gx2, dtype)
    assert torch.count_nonzero(res_gx1) == 0
    assert torch.count_nonzero(res_gx2) == 0


@pytest.mark.cdist_forward_backward
@pytest.mark.parametrize("p", [0.5, 1.5])
def test_cdist_forward_backward_partial_zero_diff(p, dtype=torch.float32):
    """p < 2 with one coordinate exactly equal while the distance stays > 0.

    diff == 0 selects the ATen guard (the same lane can also hit pow(0, p-1)
    for p < 1), which the identical-row test does not isolate because there
    the whole row -- and therefore the distance -- is zero.
    """
    batch, n1, n2 = 1, 2, 3
    res_x1 = torch.tensor(
        [[[0.0, 1.0], [2.0, 0.0]]], dtype=dtype, device=flag_gems.device
    )
    res_x2 = torch.tensor(
        [[[0.0, 2.0], [3.0, 0.0], [1.0, 1.0]]], dtype=dtype, device=flag_gems.device
    )

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_cdist = torch.cdist(ref_x1, ref_x2, p=p)
    res_cdist = ref_cdist.clone().to(flag_gems.device)
    ref_grad = torch.randn(batch, n1, n2, dtype=dtype, device=ref_cdist.device)
    res_grad = ref_grad.clone().to(flag_gems.device)

    ref_gx1 = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, p, ref_cdist)
    ref_gx2 = torch.ops.aten._cdist_backward(
        ref_grad.transpose(-2, -1), ref_x2, ref_x1, p, ref_cdist.transpose(-2, -1)
    )
    res_gx1, res_gx2 = flag_gems.cdist_forward_backward(
        res_grad, res_x1, res_x2, p, res_cdist
    )

    assert torch.isfinite(res_gx1).all()
    assert torch.isfinite(res_gx2).all()
    _assert_cdist_fb_close(res_gx1, ref_gx1, dtype, p)
    _assert_cdist_fb_close(res_gx2, ref_gx2, dtype, p)


@pytest.mark.cdist_forward_backward
# float32 only: the autograd chain exercises graph mechanics, not dtype
# coverage, which the parametrized direct tests above already provide.
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_cdist_forward_backward_autograd_chain(dtype, p):
    """Full autograd chain: FlagGems backward vs native autograd gradients.

    Builds the native cdist graph with requires_grad inputs, takes the ATen
    native _cdist_backward result as the upstream gradient, and compares the
    FlagGems kernel output against torch.autograd.grad on the same graph.
    """
    batch, n1, n2, dim = 2, 8, 6, 16
    x1 = torch.randn(
        batch, n1, dim, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    x2 = torch.randn(
        batch, n2, dim, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    ref_x1 = utils.to_reference(x1).detach().requires_grad_(True)
    ref_x2 = utils.to_reference(x2).detach().requires_grad_(True)

    ref_out = torch.cdist(ref_x1, ref_x2, p=p)
    res_out = torch.cdist(x1, x2, p=p)

    ref_grad = torch.randn(ref_out.shape, dtype=dtype, device=ref_out.device)
    res_grad = ref_grad.clone().to(flag_gems.device)

    ref_gx1, ref_gx2 = torch.autograd.grad(
        ref_out, (ref_x1, ref_x2), utils.to_reference(res_grad)
    )
    res_gx1, res_gx2 = flag_gems.cdist_forward_backward(
        res_grad, x1.detach(), x2.detach(), p, res_out.detach()
    )

    utils.gems_assert_close(res_gx1, ref_gx1, dtype)
    utils.gems_assert_close(res_gx2, ref_gx2, dtype)
