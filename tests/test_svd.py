"""Accuracy tests for torch.svd — covers all supported dtype/parameter combos.

Only float32 is supported on this platform (torch.svd CUDA limitation).
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DTYPE = torch.float32

SHAPES_2D = [(1, 1), (8, 8), (16, 16), (16, 8), (8, 32)]
SHAPES_3D = [(2, 4, 4), (4, 8, 8)]
SHAPES_4D = [(2, 3, 4, 4)]
SHAPES_EMPTY = [(0, 3), (3, 0), (2, 3, 0)]

SOME_VALUES = [True, False]
COMPUTE_UV_VALUES = [True, False]


def _reconstruct(u, s, v):
    k = s.shape[-1]
    return (u[..., :, :k] * s.unsqueeze(-2)) @ v[..., :, :k].transpose(-2, -1)


# ---------------------------------------------------------------------------
# 2D shapes — all some × compute_uv combos
# ---------------------------------------------------------------------------


@pytest.mark.svd
@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("some", SOME_VALUES)
@pytest.mark.parametrize("compute_uv", COMPUTE_UV_VALUES)
def test_svd_2d(shape, some, compute_uv):
    inp = torch.randn(shape, dtype=DTYPE, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=some, compute_uv=compute_uv)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=some, compute_uv=compute_uv)

    assert res_u.shape == ref_u.shape, f"U shape: {res_u.shape} vs {ref_u.shape}"
    assert res_s.shape == ref_s.shape, f"S shape: {res_s.shape} vs {ref_s.shape}"
    assert res_v.shape == ref_v.shape, f"V shape: {res_v.shape} vs {ref_v.shape}"

    utils.gems_assert_close(res_s, ref_s, torch.float32, equal_nan=True)

    if compute_uv and min(shape[-2], shape[-1]) > 0:
        recon = _reconstruct(res_u, res_s, res_v)
        utils.gems_assert_close(recon, ref_inp, DTYPE, equal_nan=True)


# ---------------------------------------------------------------------------
# 3D / 4D batched
# ---------------------------------------------------------------------------


@pytest.mark.svd
@pytest.mark.parametrize("shape", SHAPES_3D + SHAPES_4D)
@pytest.mark.parametrize("some", SOME_VALUES)
@pytest.mark.parametrize("compute_uv", COMPUTE_UV_VALUES)
def test_svd_batched(shape, some, compute_uv):
    inp = torch.randn(shape, dtype=DTYPE, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_u, ref_s, ref_v = torch.svd(ref_inp, some=some, compute_uv=compute_uv)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp, some=some, compute_uv=compute_uv)

    assert res_u.shape == ref_u.shape
    assert res_s.shape == ref_s.shape
    assert res_v.shape == ref_v.shape
    utils.gems_assert_close(res_s, ref_s, torch.float32, equal_nan=True)


# ---------------------------------------------------------------------------
# Empty
# ---------------------------------------------------------------------------


@pytest.mark.svd
@pytest.mark.parametrize("shape", SHAPES_EMPTY)
def test_svd_empty(shape):
    inp = torch.randn(shape, dtype=DTYPE, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_u, ref_s, ref_v = torch.svd(ref_inp)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp)

    assert res_u.shape == ref_u.shape
    assert res_s.shape == ref_s.shape
    assert res_v.shape == ref_v.shape


# ---------------------------------------------------------------------------
# Special matrices
# ---------------------------------------------------------------------------


@pytest.mark.svd
@pytest.mark.parametrize("n", [3, 8, 16])
@pytest.mark.parametrize("mat_type", ["identity", "diagonal"])
def test_svd_special_matrices(n, mat_type):
    device = flag_gems.device
    if mat_type == "identity":
        inp = torch.eye(n, dtype=DTYPE, device=device)
    else:
        inp = torch.diag(torch.arange(1, n + 1, dtype=DTYPE, device=device))

    ref_inp = utils.to_reference(inp, True)
    ref_u, ref_s, ref_v = torch.svd(ref_inp)
    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp)

    utils.gems_assert_close(res_s, ref_s, torch.float32, equal_nan=True)
    recon = _reconstruct(res_u, res_s, res_v)
    utils.gems_assert_close(recon, ref_inp, DTYPE, equal_nan=True)


# ---------------------------------------------------------------------------
# Non-contiguous
# ---------------------------------------------------------------------------


@pytest.mark.svd
@pytest.mark.parametrize("shape", [(8, 5), (3, 16, 8)])
def test_svd_non_contiguous(shape):
    inp = torch.randn(shape, dtype=DTYPE, device=flag_gems.device).transpose(-2, -1)
    ref_inp = utils.to_reference(inp, True)
    ref_u, ref_s, ref_v = torch.svd(ref_inp)

    with flag_gems.use_gems(include=["svd"]):
        res_u, res_s, res_v = torch.svd(inp)

    assert res_u.shape == ref_u.shape
    utils.gems_assert_close(res_s, ref_s, torch.float32, equal_nan=True)
