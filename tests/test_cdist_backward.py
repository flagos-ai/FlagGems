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

# (batch, n1, dim) shapes covering small/medium/large cases
CDIST_BACKWARD_SHAPES = [(2, 16, 32), (4, 32, 64), (8, 64, 128)]

# p-norm orders exercising the L1/L2/L-inf special cases and the general path
CDIST_BACKWARD_P = [1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]


def _assert_cdist_backward_close(res_out, ref_out, dtype, p):
    # The p-norm backward for p < 1 has an extremely large dynamic range in
    # fp32 (|diff|^(p-1) amplifies small differences), so even ATen CUDA vs
    # CPU disagree at the ~1e-3 relative level. Use a relaxed tolerance there.
    if p < 1.0:
        res_out = res_out.to(device=ref_out.device)
        torch.testing.assert_close(res_out, ref_out, atol=1e-2, rtol=1e-2)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cdist_backward
@pytest.mark.parametrize("shape", CDIST_BACKWARD_SHAPES)
# _cdist_backward uses intermediate fp32 accumulation; only float32 is numerically stable
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("p", [0.5] + CDIST_BACKWARD_P)
def test_cdist_backward(shape, dtype, p):
    # shape is (batch, n1, dim), n2 is separate
    batch, n1, dim = shape
    n2 = n1 // 2 + 1  # Use different n2 for variety

    res_x1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_x2 = torch.randn(batch, n2, dim, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn(batch, n1, n2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_grad = utils.to_reference(res_grad)

    # Compute cdist first
    ref_cdist = torch.cdist(ref_x1, ref_x2, p=p)
    res_cdist = ref_cdist.clone().to(flag_gems.device)

    ref_out = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, p, ref_cdist)
    # Call the FlagGems kernel directly instead of patching the ATen op with
    # flag_gems.use_gems() so the test stays in the upstream no-context style.
    res_out = flag_gems._cdist_backward(res_grad, res_x1, res_x2, p, res_cdist)

    _assert_cdist_backward_close(res_out, ref_out, dtype, p)


@pytest.mark.cdist_backward
@pytest.mark.parametrize("p", CDIST_BACKWARD_P)
def test_cdist_backward_zero_distance(p, dtype=torch.float32):
    """Guard against dist == 0 / diff == 0 lanes (identical points)."""
    batch, n1, n2, dim = 2, 5, 6, 8
    res_x1 = torch.randn(batch, n1, dim, dtype=dtype, device=flag_gems.device)
    res_x2 = torch.randn(batch, n2, dim, dtype=dtype, device=flag_gems.device)
    # Force some identical rows so that cdist contains exact zeros and the
    # corresponding diff vectors are exactly zero (dist==0 / diff==0 guards).
    res_x2[:, 0] = res_x1[:, 0]
    res_x2[:, 1] = res_x1[:, 2]
    res_grad = torch.randn(batch, n1, n2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_grad = utils.to_reference(res_grad)

    ref_cdist = torch.cdist(ref_x1, ref_x2, p=p)
    res_cdist = ref_cdist.clone().to(flag_gems.device)

    ref_out = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, p, ref_cdist)
    res_out = flag_gems._cdist_backward(res_grad, res_x1, res_x2, p, res_cdist)

    _assert_cdist_backward_close(res_out, ref_out, dtype, p)


@pytest.mark.cdist_backward
def test_cdist_backward_padded_lanes_no_nan(dtype=torch.float32):
    """Padded n2 lanes must not fold inf * 0 == NaN into the gradient.

    With BLOCK_N2=64 and n2=1 the masked lanes compute |1e10|^(p-1) * grad, i.e.
    inf * 0; before masking that NaN leaked into the reduction.
    """
    batch, n1, n2, dim = 1, 1, 1, 1
    big = 1e10
    res_x1 = torch.full((batch, n1, dim), big, dtype=dtype, device=flag_gems.device)
    res_x2 = torch.full((batch, n2, dim), big, dtype=dtype, device=flag_gems.device)
    res_grad = torch.ones(batch, n1, n2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_grad = utils.to_reference(res_grad)
    ref_cdist = torch.cdist(ref_x1, ref_x2, p=5.0)
    res_cdist = ref_cdist.clone().to(flag_gems.device)
    ref_out = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, 5.0, ref_cdist)
    res_out = flag_gems._cdist_backward(res_grad, res_x1, res_x2, 5.0, res_cdist)

    assert torch.isfinite(res_out).all(), f"non-finite gradient: {res_out}"
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cdist_backward
def test_cdist_backward_p_zero(dtype=torch.float32):
    """p == 0 (zero-norm distance) has an identically zero gradient."""
    batch, n1, n2, dim = 2, 4, 3, 5
    res_x1 = torch.randn(batch, n1, dim, dtype=dtype, device=flag_gems.device)
    res_x2 = torch.randn(batch, n2, dim, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn(batch, n1, n2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_grad = utils.to_reference(res_grad)
    ref_cdist = torch.cdist(ref_x1, ref_x2, p=0.0)
    res_cdist = ref_cdist.clone().to(flag_gems.device)
    ref_out = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, 0.0, ref_cdist)
    res_out = flag_gems._cdist_backward(res_grad, res_x1, res_x2, 0.0, res_cdist)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert torch.count_nonzero(res_out) == 0


@pytest.mark.cdist_backward
@pytest.mark.parametrize("p", [0.5, 1.5])
def test_cdist_backward_partial_zero_diff(p, dtype=torch.float32):
    """p < 2 with one coordinate exactly equal while the distance stays > 0.

    diff == 0 selects the ATen guard (the same lane can also hit pow(0, p-1) for
    p < 1), which the identical-row test does not isolate because there the
    whole row -- and therefore the distance -- is zero.
    """
    batch, n1, n2 = 1, 2, 3
    res_x1 = torch.tensor(
        [[[0.0, 1.0], [2.0, 0.0]]], dtype=dtype, device=flag_gems.device
    )
    res_x2 = torch.tensor(
        [[[0.0, 2.0], [3.0, 0.0], [1.0, 1.0]]], dtype=dtype, device=flag_gems.device
    )
    res_grad = torch.randn(batch, n1, n2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(res_x1)
    ref_x2 = utils.to_reference(res_x2)
    ref_grad = utils.to_reference(res_grad)
    ref_cdist = torch.cdist(ref_x1, ref_x2, p=p)
    res_cdist = ref_cdist.clone().to(flag_gems.device)
    ref_out = torch.ops.aten._cdist_backward(ref_grad, ref_x1, ref_x2, p, ref_cdist)
    res_out = flag_gems._cdist_backward(res_grad, res_x1, res_x2, p, res_cdist)

    assert torch.isfinite(res_out).all()
    _assert_cdist_backward_close(res_out, ref_out, dtype, p)
