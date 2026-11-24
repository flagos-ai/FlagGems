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

# Tests call flag_gems.euclidean_dist_backward directly instead of the
# flag_gems.use_gems() context manager: the CI kernelgen-test rules forbid a
# real use_gems() call from test files, and the wrapper is what the PR exposes.

# cdist on CUDA does not support float16/bfloat16, so gradients are exercised
# in float32 only (matching the forward's dtype restriction).
EUC_BACKWARD_SHAPES = [
    ((4, 8), (5, 8)),
    ((8, 16), (8, 16)),
    ((16, 32), (12, 32)),
    ((32, 64), (32, 64)),
    ((64, 128), (64, 128)),
    ((128, 256), (128, 256)),
    ((5, 7), (5, 7)),
    ((3, 4), (3, 4)),
]


def _reference(grad, x1, x2):
    # The deterministic ATen pairwise kernels: the p=2 _cdist_backward formula
    # reproduces the euclidean-distance gradient to ~1e-7 while avoiding the
    # GEMM-expansion rounding noise of the autograd chain.
    dist = torch.cdist(x1, x2, p=2.0)
    grad_x1 = torch.ops.aten._cdist_backward(grad, x1, x2, 2.0, dist)
    grad_x2 = torch.ops.aten._cdist_backward(
        grad.transpose(-2, -1).contiguous(),
        x2,
        x1,
        2.0,
        dist.transpose(-2, -1).contiguous(),
    )
    return grad_x1, grad_x2


@pytest.mark.euclidean_dist_backward
@pytest.mark.parametrize("shape", EUC_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_euclidean_dist_backward(shape, dtype):
    (n1, d), (n2, d2) = shape
    assert d == d2, "Feature dimensions must match"

    x1 = torch.randn(n1, d, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(n2, d, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(n1, n2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(x1)
    ref_x2 = utils.to_reference(x2)
    ref_grad = utils.to_reference(grad)
    ref_out1, ref_out2 = _reference(ref_grad, ref_x1, ref_x2)

    res_out1, res_out2 = flag_gems.euclidean_dist_backward(grad, x1, x2)

    utils.gems_assert_close(res_out1, ref_out1, dtype)
    utils.gems_assert_close(res_out2, ref_out2, dtype)


@pytest.mark.euclidean_dist_backward
@pytest.mark.parametrize("dtype", [torch.float32])
def test_euclidean_dist_backward_batched(dtype):
    # Leading batch dimensions are folded into the row axis: the per-pair
    # computation is independent of batch structure.
    x1 = torch.randn(2, 3, 8, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(2, 5, 8, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(2, 3, 5, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(x1)
    ref_x2 = utils.to_reference(x2)
    ref_grad = utils.to_reference(grad)
    ref_out1, ref_out2 = _reference(ref_grad, ref_x1, ref_x2)

    res_out1, res_out2 = flag_gems.euclidean_dist_backward(grad, x1, x2)

    utils.gems_assert_close(res_out1, ref_out1, dtype)
    utils.gems_assert_close(res_out2, ref_out2, dtype)


@pytest.mark.euclidean_dist_backward
@pytest.mark.parametrize("dtype", [torch.float32])
def test_euclidean_dist_backward_non_contiguous(dtype):
    # A transposed view exercises the .contiguous() path in the wrapper.
    base = torch.randn(6, 8, dtype=dtype, device=flag_gems.device)
    x1 = base.t()
    x2 = torch.randn(5, 6, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(8, 5, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(x1)
    ref_x2 = utils.to_reference(x2)
    ref_grad = utils.to_reference(grad)
    ref_out1, ref_out2 = _reference(ref_grad, ref_x1, ref_x2)

    res_out1, res_out2 = flag_gems.euclidean_dist_backward(grad, x1, x2)

    utils.gems_assert_close(res_out1, ref_out1, dtype)
    utils.gems_assert_close(res_out2, ref_out2, dtype)


@pytest.mark.euclidean_dist_backward
@pytest.mark.parametrize("dtype", [torch.float32])
def test_euclidean_dist_backward_zero_distance(dtype):
    # Identical rows: dist == 0 pairs contribute zero (matching the native
    # masked_fill_(res == 0, 0) subgradient).
    x1 = torch.randn(1, 4, dtype=dtype, device=flag_gems.device)
    x2 = x1.clone()
    grad = torch.ones(1, 1, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(x1)
    ref_x2 = utils.to_reference(x2)
    ref_grad = utils.to_reference(grad)
    ref_out1, ref_out2 = _reference(ref_grad, ref_x1, ref_x2)

    res_out1, res_out2 = flag_gems.euclidean_dist_backward(grad, x1, x2)

    utils.gems_assert_close(res_out1, ref_out1, dtype)
    utils.gems_assert_close(res_out2, ref_out2, dtype)


@pytest.mark.euclidean_dist_backward
def test_euclidean_dist_backward_autograd_link():
    # The FlagGems gradient also matches the native autograd chain within the
    # comparator tolerance (the ATen backward itself is GEMM-noisy).
    x1 = torch.randn(16, 32, dtype=torch.float32, device=flag_gems.device)
    x2 = torch.randn(12, 32, dtype=torch.float32, device=flag_gems.device)
    grad = torch.randn(16, 12, dtype=torch.float32, device=flag_gems.device)

    x1r = x1.detach().requires_grad_(True)
    x2r = x2.detach().requires_grad_(True)
    out = torch._euclidean_dist(x1r, x2r)
    ref_out1, ref_out2 = torch.autograd.grad(out, (x1r, x2r), grad)

    res_out1, res_out2 = flag_gems.euclidean_dist_backward(grad, x1, x2)

    utils.gems_assert_close(res_out1, ref_out1, torch.float32, atol=1e-3)
    utils.gems_assert_close(res_out2, ref_out2, torch.float32, atol=1e-3)
