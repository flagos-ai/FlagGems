import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for linalg_eigh (square matrices) — linalg_eigh only supports float32/float64
EIG_SHAPES = [(2, 2), (3, 3), (5, 5), (8, 8), (16, 16), (32, 32)]
# Batch shapes for linalg_eigh — square matrices in batch
EIG_BATCH_SHAPES = [(2, 2, 2), (4, 3, 3), (1, 8, 8)]


def make_symmetric_matrix(shape, dtype, device):
    """Create a symmetric matrix for eigendecomposition."""
    if len(shape) >= 2 and shape[-1] == shape[-2]:
        A = torch.randn(shape, dtype=dtype, device=device)
        A = (A + A.transpose(-2, -1)) / 2
        return A
    else:
        raise ValueError(f"Expected square matrix shape, got {shape}")


@pytest.mark.linalg_eigh
@pytest.mark.parametrize("shape", EIG_SHAPES)
# linalg_eigh only supports float32/float64 on GPU; fp16/bf16 not supported
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh(shape, dtype):
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Compare eigenvalues
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)


@pytest.mark.linalg_eigh
@pytest.mark.parametrize("shape", EIG_BATCH_SHAPES)
# linalg_eigh only supports float32/float64 on GPU; fp16/bf16 not supported
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigh_batch(shape, dtype):
    inp = make_symmetric_matrix(shape, dtype, flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.linalg.eigh(ref_inp)

    with flag_gems.use_gems():
        res_out = torch.linalg.eigh(inp)

    # Compare eigenvalues
    utils.gems_assert_close(res_out[0], ref_out[0], dtype)
