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

POLAR_SHAPES = (
    [(8, 4)]
    if cfg.QUICK_MODE
    else [
        (3, 2),
        (8, 4),
        (16, 8),
        (2, 8, 4),
        (8, 16, 8),
    ]
)
ATOL = 1e-2


def _reference_linalg_polar(inp):
    # There is no ATen polar primitive. Keep this independent SVD composition
    # on CPU even when the test's default reference device is the accelerator.
    inp = inp.detach().cpu()
    try:
        svd_u, singular_values, vh = torch.linalg.svd(inp, full_matrices=False)
    except RuntimeError as error:
        if str(error) != "svd: LAPACK library not found in compilation":
            raise
        import numpy as np

        u, values, v = np.linalg.svd(inp.numpy(), full_matrices=False)
        svd_u, singular_values, vh = (torch.from_numpy(x) for x in (u, values, v))
    polar_u = svd_u @ vh
    polar_h = vh.mH @ (singular_values.unsqueeze(-1) * vh)
    return polar_u, 0.5 * (polar_h + polar_h.mH)


def _assert_output_metadata(inp, polar_u, polar_h):
    for result, shape in (
        (polar_u, inp.shape),
        (polar_h, (*inp.shape[:-2], inp.shape[-1], inp.shape[-1])),
    ):
        assert result.device == inp.device
        assert result.dtype == inp.dtype
        assert result.shape == shape


def _empty_with_dtype(shape, dtype, device):
    result = torch.empty(shape, dtype=dtype, device=device)
    if result.dtype != dtype:
        # Some device factories silently narrow Double. The negative case must
        # still present the requested dtype, rather than exercise a Float input.
        result = (
            torch.empty(shape, dtype=dtype)
            .view(torch.uint8)
            .to(device)
            .view(dtype)
            .reshape(shape)
        )
    assert result.dtype == dtype
    return result


def _assert_polar_properties(inp, polar_u, polar_h):
    # All callers check the original outputs' metadata before CPU observation.
    inp = inp.detach().cpu()
    polar_u = polar_u.detach().cpu()
    polar_h = polar_h.detach().cpu()
    utils.gems_assert_close(polar_u @ polar_h, inp, torch.float32, atol=ATOL)
    utils.gems_assert_close(polar_h, polar_h.mT, torch.float32, atol=ATOL)
    gram = polar_u.mT @ polar_u
    eye = torch.eye(polar_u.shape[-1], dtype=polar_u.dtype)
    utils.gems_assert_close(gram, eye.expand_as(gram), torch.float32, atol=ATOL)


@pytest.mark.linalg_polar
@pytest.mark.parametrize("shape", POLAR_SHAPES)
def test_linalg_polar(shape):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp_before = inp.clone()
    ref_inp = inp.detach().cpu().to(torch.float64)
    _, ref_h = _reference_linalg_polar(ref_inp)

    polar_u, polar_h = flag_gems.linalg_polar(inp)
    _assert_output_metadata(inp, polar_u, polar_h)

    assert polar_u.shape == inp.shape
    assert polar_h.shape == (*inp.shape[:-2], inp.shape[-1], inp.shape[-1])
    assert polar_u.is_contiguous()
    assert polar_h.is_contiguous()
    assert polar_u is not inp
    assert inp.device == inp_before.device
    assert inp.dtype == inp_before.dtype
    utils.gems_assert_equal(inp.cpu(), inp_before.cpu())
    utils.gems_assert_close(polar_h.cpu(), ref_h, torch.float32, atol=ATOL)
    _assert_polar_properties(ref_inp, polar_u, polar_h)


@pytest.mark.linalg_polar
def test_linalg_polar_noncontiguous():
    torch.manual_seed(0)
    inp = torch.randn((4, 16), device=flag_gems.device).mT
    assert not inp.is_contiguous()
    ref_inp = inp.detach().cpu().to(torch.float64)

    polar_u, polar_h = flag_gems.linalg_polar(inp)
    _assert_output_metadata(inp, polar_u, polar_h)

    _assert_polar_properties(ref_inp, polar_u, polar_h)


@pytest.mark.linalg_polar
@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 5, 3)])
def test_linalg_polar_empty(shape):
    inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = inp.detach().cpu().to(torch.float64)
    ref_u, ref_h = _reference_linalg_polar(ref_inp)

    polar_u, polar_h = flag_gems.linalg_polar(inp)
    _assert_output_metadata(inp, polar_u, polar_h)

    assert polar_u.shape == ref_u.shape
    assert polar_h.shape == ref_h.shape
    assert polar_u.dtype == inp.dtype
    assert polar_h.dtype == inp.dtype
    assert polar_u.is_contiguous()
    assert polar_h.is_contiguous()


@pytest.mark.linalg_polar
def test_linalg_polar_rank_deficient():
    inp = torch.zeros((8, 4), dtype=torch.float32, device=flag_gems.device)
    ref_inp = inp.detach().cpu().to(torch.float64)

    polar_u, polar_h = flag_gems.linalg_polar(inp)
    _assert_output_metadata(inp, polar_u, polar_h)

    utils.gems_assert_equal(
        polar_h.cpu(), torch.zeros_like(ref_inp[:4], dtype=inp.dtype)
    )
    utils.gems_assert_equal(polar_u.cpu() @ polar_h.cpu(), ref_inp.to(inp.dtype))


@pytest.mark.linalg_polar
@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((3,), torch.float32),
        ((3, 5), torch.float32),
        ((5, 3), torch.float64),
    ],
)
def test_linalg_polar_invalid_input(shape, dtype):
    inp = _empty_with_dtype(shape, dtype, flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.linalg_polar(inp)


@pytest.mark.linalg_polar
def test_linalg_polar_autograd_is_explicitly_unsupported():
    torch.manual_seed(0)
    inp = torch.randn((8, 4), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_(True)

    polar_u, polar_h = flag_gems.linalg_polar(inp)
    _assert_output_metadata(inp, polar_u, polar_h)

    assert not polar_u.requires_grad
    assert not polar_h.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(polar_h.square().sum(), inp)
