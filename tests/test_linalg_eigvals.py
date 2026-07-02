import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_eigvals
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (5, 5), (10, 10), (20, 20)])
# _linalg_eigvals requires float32 for cuSOLVER eigenvalue computation
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals(shape, dtype):
    """Test _linalg_eigvals accuracy against PyTorch reference."""
    # Create a square matrix
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._linalg_eigvals.default(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._linalg_eigvals.default(inp)

    # Compare complex eigenvalues - use the output dtype for comparison
    # For float32 input, output is complex64
    utils.gems_assert_close(res_out, ref_out, res_out.dtype)
