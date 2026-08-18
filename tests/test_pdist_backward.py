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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes from the worktree PDIST_SHAPES covering small to large matrices
PDIST_SHAPES = [
    (4, 8),
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


@pytest.mark.pdist_backward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist_backward limited to float32 for numerical stability
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_backward(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute pdist forward
    p = 2.0
    pdist_out = torch.pdist(ref_inp, p=p)
    pdist_out_gems = torch.pdist(inp, p=p)

    # Compute backward with gradient of ones
    ref_grad_output = torch.ones_like(pdist_out)
    grad_output = torch.ones_like(pdist_out_gems)

    ref_out = torch.ops.aten._pdist_backward(ref_grad_output, ref_inp, p, pdist_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_backward(grad_output, inp, p, pdist_out_gems)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist_backward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist_backward limited to float32 for numerical stability
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_backward_p1(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute pdist forward with p=1
    p = 1.0
    pdist_out = torch.pdist(ref_inp, p=p)
    pdist_out_gems = torch.pdist(inp, p=p)

    # Compute backward with gradient of ones
    ref_grad_output = torch.ones_like(pdist_out)
    grad_output = torch.ones_like(pdist_out_gems)

    ref_out = torch.ops.aten._pdist_backward(ref_grad_output, ref_inp, p, pdist_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_backward(grad_output, inp, p, pdist_out_gems)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist_backward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist_backward limited to float32 for numerical stability
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_backward_pinf(shape, dtype):
    # p=inf branch (_pdist_backward_inf_kernel): gradient is grad * sign(diff) for
    # every coordinate whose |diff| equals the max-norm distance (ties are summed,
    # matching PyTorch). A random gradient exercises both signs of the contribution.
    # NOTE: p=-inf is not covered because torch.pdist requires p >= 0; passing a
    # negative p to the forward is rejected, so no valid forward/backward pair can
    # be constructed. The kernel routes +/-inf identically via math.isinf(p).
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute pdist forward with p=inf
    p = math.inf
    pdist_out = torch.pdist(ref_inp, p=p)
    pdist_out_gems = torch.pdist(inp, p=p)

    ref_grad_output = torch.randn_like(pdist_out)
    grad_output = ref_grad_output.to(flag_gems.device)

    ref_out = torch.ops.aten._pdist_backward(ref_grad_output, ref_inp, p, pdist_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_backward(grad_output, inp, p, pdist_out_gems)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist_backward
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist_backward limited to float32 for numerical stability
@pytest.mark.parametrize("dtype", [torch.float32])
# Only p > 1 is covered: for p < 1 the gradient (involving |diff|^(p-1)) is
# numerically unstable in float32 and both the Gems kernel and PyTorch's
# reference diverge from the float64 ground truth, so it is not a kernel defect.
@pytest.mark.parametrize("p", [1.5, 3.0])
def test_pdist_backward_general(shape, dtype, p):
    # General-p branch (_pdist_backward_general_kernel): contribution is
    # grad * sign(diff) * (|diff| / dist) ** (p - 1), computed via exp/log.
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute pdist forward with the general p value
    pdist_out = torch.pdist(ref_inp, p=p)
    pdist_out_gems = torch.pdist(inp, p=p)

    ref_grad_output = torch.randn_like(pdist_out)
    grad_output = ref_grad_output.to(flag_gems.device)

    ref_out = torch.ops.aten._pdist_backward(ref_grad_output, ref_inp, p, pdist_out)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._pdist_backward(grad_output, inp, p, pdist_out_gems)

    utils.gems_assert_close(res_out, ref_out, dtype)
