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
from . import conftest as cfg


DIRICHLET_GRAD_DTYPES = [torch.float32, torch.float64]
_ALPHA_RANGES = [(0.5, 10.0), (1.0, 50.0), (0.2, 100.0)]
_SHAPES = [
    (*batch, categories)
    for batch in ((), (8,), (64,), (20, 320))
    for categories in (2, 4, 15, 32)
] + [(20, 32)]


def _run_dirichlet_grad(x, alpha, total):
    candidate = getattr(flag_gems, "dirichlet_grad", None)
    if candidate is None:
        candidate = flag_gems._dirichlet_grad
    return candidate(x, alpha, total)


def _make_inputs(shape, dtype, alpha_range):
    torch.manual_seed(0)
    lo, hi = alpha_range
    alpha = torch.rand(shape, dtype=dtype, device=flag_gems.device) * (hi - lo) + lo
    total = alpha.sum(-1, keepdim=True).expand_as(alpha).contiguous()
    gamma_shape = alpha if dtype == torch.float64 else alpha.float()
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float64:
        gamma_shape = gamma_shape.cpu()
    samples = torch.distributions.Gamma(
        gamma_shape, torch.ones_like(gamma_shape)
    ).sample().to(device=alpha.device, dtype=dtype)
    x = samples / samples.sum(-1, keepdim=True)
    return x, alpha, total


@pytest.mark.dirichlet_grad
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", DIRICHLET_GRAD_DTYPES)
@pytest.mark.parametrize("alpha_range", _ALPHA_RANGES)
def test_dirichlet_grad_regimes(shape, dtype, alpha_range):
    x, alpha, total = _make_inputs(shape, dtype, alpha_range)
    x_before = x.clone()
    alpha_before = alpha.clone()
    total_before = total.clone()
    # Keep the original input dtype on MUSA; the generic upcast can narrow
    # FP64 when the backend advertises incomplete FP64 support.
    ref_x, ref_alpha, ref_total = (
        utils.to_reference(value, flag_gems.vendor_name != "mthreads")
        for value in (x, alpha, total)
    )

    reference = torch.ops.aten._dirichlet_grad(ref_x, ref_alpha, ref_total)
    result = _run_dirichlet_grad(x, alpha, total)

    assert result.device == x.device
    assert result.shape == x.shape
    assert result.dtype == dtype
    utils.gems_assert_close(result, reference, dtype, atol=2e-3)
    for current, prior in ((x, x_before), (alpha, alpha_before), (total, total_before)):
        if flag_gems.vendor_name == "mthreads" and cfg.TO_CPU:
            current, prior = current.cpu(), prior.cpu()
        utils.gems_assert_equal(current, prior)


@pytest.mark.dirichlet_grad
def test_dirichlet_grad_rejects_nonbroadcastable_shape():
    x = torch.full((8, 4), 0.25, device=flag_gems.device)
    alpha = torch.ones((8, 4), device=flag_gems.device)
    total = torch.ones((3,), device=flag_gems.device)

    with pytest.raises(RuntimeError):
        _run_dirichlet_grad(x, alpha, total)
