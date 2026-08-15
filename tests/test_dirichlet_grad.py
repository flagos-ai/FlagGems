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

# `_dirichlet_grad` is the reparameterized-gradient of the Dirichlet
# distribution.  The CUDA (and CPU) reference dispatch in PyTorch only
# implements the AT_DISPATCH_FLOATING_TYPES path (float32 / float64) -- half
# and bfloat16 are *not* supported -- so we restrict the test to those two
# dtypes (see ATen/native/cuda/Distributions.cu).
# Only float32/float64 are supported by the ATen CUDA dispatcher; matching that.
DIRICHLET_GRAD_DTYPES = [torch.float32, torch.float64]

# Concentration ranges.  The kernel switches between small / large / mid /
# rational sub-branches depending on `alpha`, `total` and the sampled `x`, so
# covering a few concentration regimes exercises all of them.
DIRICHLET_ALPHA_RANGES = [(0.5, 10.0), (1.0, 50.0), (0.2, 100.0)]

# Number of categories in the simplex.  The branch selection depends on
# `total*x*(1-x)` and `total`, so vary the category count as well.
DIRICHLET_KS = [2, 4, 15, 32]

# PyTorch's own `test_dirichlet_multivariate` checks the gradient with
# `atol=0.002, rtol=0`; the saddle-point sub-branch of the mid regime is
# numerically delicate, so we use the same order of tolerance.
DIRICHLET_GRAD_ATOL = 1e-3


def _dirichlet_inputs(shape, dtype, alpha_range, seed):
    """Build a Dirichlet-distributed ``x`` and matching ``alpha`` / ``total``.

    ``alpha`` is drawn uniformly in ``alpha_range``.  ``x`` is sampled from a
    Gamma distribution with shape ``alpha`` (keeping the Gamma draws in the
    input dtype so the rounding matches what a user would feed the op) and
    then normalized along the last dimension to lie on the simplex -- this is
    the canonical way `torch.distributions.Dirichlet` constructs samples.
    ``total`` is the sum of the concentrations broadcast back to ``alpha``'s
    shape, which is exactly the third argument the ATen op expects.
    """
    utils.init_seed(seed)
    lo, hi = alpha_range
    alpha = torch.rand(*shape, dtype=dtype, device=flag_gems.device) * (hi - lo) + lo
    total = alpha.sum(-1, keepdim=True).expand_as(alpha).contiguous()
    # Gamma needs a float32-or-wider rate/shape on CUDA; sample then cast back.
    gamma_shape = alpha if dtype == torch.float64 else alpha.float()
    gamma = torch.distributions.Gamma(gamma_shape, torch.ones_like(gamma_shape))
    x = gamma.sample().to(dtype)
    x = x / x.sum(-1, keepdim=True)
    return x, alpha, total


@pytest.mark.dirichlet_grad
@pytest.mark.parametrize("dtype", DIRICHLET_GRAD_DTYPES)
@pytest.mark.parametrize("alpha_range", DIRICHLET_ALPHA_RANGES)
@pytest.mark.parametrize("k", DIRICHLET_KS)
@pytest.mark.parametrize("shape_kind", ["flat", "rows", "batched", "deep"])
def test_dirichlet_grad(shape_kind, dtype, alpha_range, k):
    shape = {"flat": (k,), "rows": (8, k), "batched": (64, k), "deep": (20, 320, k)}[
        shape_kind
    ]
    x, alpha, total = _dirichlet_inputs(shape, dtype, alpha_range, seed=k)
    ref_x = utils.to_reference(x.clone())
    ref_alpha = utils.to_reference(alpha)
    ref_total = utils.to_reference(total)

    ref_out = torch.ops.aten._dirichlet_grad(ref_x, ref_alpha, ref_total)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._dirichlet_grad(x, alpha, total)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=DIRICHLET_GRAD_ATOL)
