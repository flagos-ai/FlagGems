import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes for sym_stride - covering various tensor dimensionalities
SYM_STRIDE_SHAPES = [(2, 3), (10, 20, 30), (5, 10), (100,), (1, 2, 3, 4)]


@pytest.mark.sym_stride
@pytest.mark.parametrize("shape", SYM_STRIDE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sym_stride(shape, dtype):
    """Test sym_stride operator accuracy."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.sym_stride(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.sym_stride(inp)

    # Compare stride results (convert to tensors for gems_assert_equal)
    utils.gems_assert_equal(torch.tensor(res_out), torch.tensor(ref_out))
