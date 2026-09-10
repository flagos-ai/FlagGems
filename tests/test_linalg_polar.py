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

if cfg.QUICK_MODE:
    POLAR_SHAPES = [(8, 4)]
else:
    POLAR_SHAPES = [
        (3, 2),
        (8, 4),
        (16, 8),
        (2, 8, 4),
        (8, 16, 8),
    ]

ATOL = 1e-2


def _reference_linalg_polar(inp):
    svd_U, singular_values, Vh = torch.linalg.svd(inp, full_matrices=False)
    polar_U = svd_U @ Vh
    polar_H = Vh.mH @ (singular_values.unsqueeze(-1) * Vh)
    return polar_U, 0.5 * (polar_H + polar_H.mH)


def _assert_polar_properties(inp, U, H):
    reconstructed = U @ H
    utils.gems_assert_close(reconstructed, inp, torch.float32, atol=ATOL)

    symmetric = utils.to_reference(H.mT, False)
    utils.gems_assert_close(H, symmetric, torch.float32, atol=ATOL)

    gram = U.mT @ U
    eye = torch.eye(U.shape[-1], dtype=U.dtype, device=flag_gems.device)
    expected = utils.to_reference(eye.expand_as(gram), False)
    utils.gems_assert_close(gram, expected, torch.float32, atol=ATOL)


@pytest.mark.linalg_polar
@pytest.mark.parametrize("shape", POLAR_SHAPES)
def test_linalg_polar(shape):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    _, ref_H = _reference_linalg_polar(ref_inp)

    result_U, result_H = flag_gems.linalg_polar(inp)

    assert result_U.shape == inp.shape
    assert result_H.shape == (*inp.shape[:-2], inp.shape[-1], inp.shape[-1])
    assert result_U.is_contiguous()
    assert result_H.is_contiguous()
    utils.gems_assert_close(result_H, ref_H, torch.float32, atol=ATOL)
    _assert_polar_properties(ref_inp, result_U, result_H)


@pytest.mark.linalg_polar
def test_linalg_polar_public_api():
    inp = torch.randn((8, 4), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    result_U, result_H = flag_gems.linalg_polar(inp)

    _assert_polar_properties(ref_inp, result_U, result_H)


@pytest.mark.linalg_polar
def test_linalg_polar_dispatcher():
    if not hasattr(torch.ops.aten, "linalg_polar"):
        pytest.skip("aten.linalg_polar is unavailable in this PyTorch version")

    inp = torch.randn((8, 4), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    result_U, result_H = flag_gems.linalg_polar(inp)

    _assert_polar_properties(ref_inp, result_U, result_H)


@pytest.mark.linalg_polar
def test_linalg_polar_noncontiguous():
    inp = torch.randn((4, 16), device=flag_gems.device).mT
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp, False)

    result_U, result_H = flag_gems.linalg_polar(inp)

    _assert_polar_properties(ref_inp, result_U, result_H)


@pytest.mark.linalg_polar
@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 5, 3)])
def test_linalg_polar_empty(shape):
    inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    ref_U, ref_H = _reference_linalg_polar(ref_inp)
    result_U, result_H = flag_gems.linalg_polar(inp)

    assert result_U.shape == ref_U.shape
    assert result_H.shape == ref_H.shape
    assert result_U.is_contiguous()
    assert result_H.is_contiguous()


@pytest.mark.linalg_polar
def test_linalg_polar_rank_deficient():
    inp = torch.zeros((8, 4), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    result_U, result_H = flag_gems.linalg_polar(inp)

    utils.gems_assert_equal(result_H, torch.zeros_like(ref_inp[:4]))
    utils.gems_assert_equal(result_U @ result_H, ref_inp)


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
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.linalg_polar(inp)


@pytest.mark.linalg_polar_out
@pytest.mark.parametrize("shape", POLAR_SHAPES)
def test_linalg_polar_out(shape):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    _, ref_H = _reference_linalg_polar(ref_inp)

    out_U = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    out_H = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    result_U, result_H = flag_gems.linalg_polar_out(inp, U=out_U, H=out_H)

    assert result_U is out_U
    assert result_H is out_H
    utils.gems_assert_close(result_H, ref_H, torch.float32, atol=ATOL)
    _assert_polar_properties(ref_inp, result_U, result_H)


@pytest.mark.linalg_polar_out
def test_linalg_polar_out_noncontiguous_buffers():
    inp = torch.randn((8, 4), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    out_U = torch.empty((4, 8), device=flag_gems.device).mT
    out_H = torch.empty((4, 4), device=flag_gems.device).mT
    U_stride = out_U.stride()
    H_stride = out_H.stride()

    result_U, result_H = flag_gems.linalg_polar_out(inp, U=out_U, H=out_H)

    assert result_U is out_U
    assert result_H is out_H
    assert result_U.stride() == U_stride
    assert result_H.stride() == H_stride
    _assert_polar_properties(ref_inp, result_U, result_H)
