import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_system(a_shape, b_ndim, dtype):
    """Build a well-conditioned tensorsolve system.

    The flattened (m, m) matrix is diagonally dominant so it stays invertible.
    """
    A = torch.randn(a_shape, dtype=dtype, device=flag_gems.device)
    m = 1
    for d in a_shape[:b_ndim]:
        m *= d
    A_flat = A.reshape(m, m)
    A_flat = A_flat + torch.eye(m, dtype=dtype, device=flag_gems.device) * m
    A = A_flat.reshape(a_shape)
    B = torch.randn(a_shape[:b_ndim], dtype=dtype, device=flag_gems.device)
    return A, B


@pytest.mark.linalg_tensorsolve
@pytest.mark.parametrize(
    "a_shape, b_ndim",
    [
        ((4, 4), 1),
        ((6, 2, 3), 1),
        ((6, 4, 2, 3, 4), 2),
        ((2, 3, 2, 3), 2),
        ((8, 2, 4), 1),
    ],
)
# linalg.tensorsolve only supports float32 and float64 in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_tensorsolve(a_shape, b_ndim, dtype):
    A, B = _make_system(a_shape, b_ndim, dtype)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.tensorsolve(ref_A, ref_B)

    with flag_gems.use_gems():
        res_out = torch.linalg.tensorsolve(A, B)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.linalg_tensorsolve
@pytest.mark.parametrize(
    "a_shape, b_shape, dims",
    [
        ((6, 4, 4, 3, 2), (4, 3, 2), (0, 2)),
        ((2, 6, 3), (2, 3), (1,)),
    ],
)
# linalg.tensorsolve only supports float32 and float64 in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_tensorsolve_dims(a_shape, b_shape, dims, dtype):
    A = torch.randn(a_shape, dtype=dtype, device=flag_gems.device)
    # Make the moved-and-flattened matrix diagonally dominant for stability.
    dest = tuple(range(len(dims) - A.ndim + 1, 0))
    A_moved = torch.movedim(A, dims, dest)
    m = 1
    for d in b_shape:
        m *= d
    A_flat = A_moved.reshape(m, m)
    A_flat = A_flat + torch.eye(m, dtype=dtype, device=flag_gems.device) * m
    A = torch.movedim(A_flat.reshape(A_moved.shape), dest, dims)
    B = torch.randn(b_shape, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.tensorsolve(ref_A, ref_B, dims=dims)

    with flag_gems.use_gems():
        res_out = torch.linalg.tensorsolve(A, B, dims=dims)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.linalg_tensorsolve_out
@pytest.mark.parametrize(
    "a_shape, b_ndim",
    [
        ((4, 4), 1),
        ((6, 4, 2, 3, 4), 2),
    ],
)
# linalg.tensorsolve only supports float32 and float64 in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_tensorsolve_out(a_shape, b_ndim, dtype):
    A, B = _make_system(a_shape, b_ndim, dtype)

    ref_A = utils.to_reference(A)
    ref_B = utils.to_reference(B)
    ref_out = torch.linalg.tensorsolve(ref_A, ref_B)

    out = torch.empty(a_shape[b_ndim:], dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.linalg.tensorsolve(A, B, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)
