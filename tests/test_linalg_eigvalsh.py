import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_symmetric(shape, dtype):
    # Build a symmetric matrix A = (B + B^T) / 2 so eigvalsh is well-defined.
    B = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return (B + B.transpose(-2, -1)) / 2


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)])
# eigvalsh only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigvalsh(shape, dtype):
    A = _make_symmetric(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.eigvalsh(ref_A)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_eigvalsh(A)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8), (16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigvalsh_upper(shape, dtype):
    A = _make_symmetric(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.eigvalsh(ref_A, UPLO="U")

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_eigvalsh(A, UPLO="U")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.linalg_eigvalsh
@pytest.mark.parametrize("shape", [(2, 4, 4), (3, 8, 8), (2, 3, 16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_eigvalsh_batch(shape, dtype):
    A = _make_symmetric(shape, dtype)

    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.eigvalsh(ref_A)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.linalg_eigvalsh(A)

    utils.gems_assert_close(res_out, ref_out, dtype)
