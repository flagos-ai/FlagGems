import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SVD_SHAPES = [
    (4, 4),
    (4, 8),
    (8, 4),
]


def _make_well_conditioned_input(shape, dtype, batch_size=None):
    m, n = shape
    row = torch.arange(m, device=flag_gems.device, dtype=torch.float32).reshape(m, 1)
    col = torch.arange(n, device=flag_gems.device, dtype=torch.float32).reshape(1, n)
    A = torch.sin((row + 1) * (col + 1) * 0.37)
    A += 0.25 * torch.cos((row + 2) * (col + 1) * 0.19)

    diag = torch.arange(min(m, n), device=flag_gems.device)
    A[diag, diag] += torch.linspace(2.0, 3.0, min(m, n), device=flag_gems.device)

    if batch_size is not None:
        A = A.unsqueeze(0).repeat(batch_size, 1, 1)
        offsets = torch.arange(
            batch_size, device=flag_gems.device, dtype=torch.float32
        ).reshape(batch_size, 1, 1)
        A = A + offsets * 0.05

    return A.to(dtype)


def _reference_svdvals(A, dtype):
    ref_A = utils.to_reference(A)
    if dtype in (torch.bfloat16, torch.float16):
        ref_A = ref_A.to(torch.float32)
    return torch.linalg.svdvals(ref_A)


@pytest.mark.linalg_svdvals
@pytest.mark.parametrize("shape", SVD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_svdvals(shape, dtype):
    A = _make_well_conditioned_input(shape, dtype)
    ref_out = _reference_svdvals(A, dtype)

    with flag_gems.use_gems():
        res_out = torch.linalg.svdvals(A)

    utils.gems_assert_close(res_out, ref_out, res_out.dtype, atol=2e-3)


@pytest.mark.linalg_svdvals
@pytest.mark.parametrize("shape", SVD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_svdvals_batched(shape, dtype):
    m, n = shape
    batch_size = 4

    A = _make_well_conditioned_input((m, n), dtype, batch_size=batch_size)
    ref_out = _reference_svdvals(A, dtype)

    with flag_gems.use_gems():
        res_out = torch.linalg.svdvals(A)

    utils.gems_assert_close(res_out, ref_out, res_out.dtype, atol=2e-3)


@pytest.mark.linalg_svdvals
def test_linalg_svdvals_driver_unsupported():
    A = torch.randn((4, 4), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(NotImplementedError):
        with flag_gems.use_gems():
            torch.linalg.svdvals(A, driver="gesvd")
