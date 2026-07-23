import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes for sym_size operator - returns tensor shape as a list
SHAPES_FOR_SYMSIZE = [
    (2, 3),
    (128, 256),
    (512, 512),
    (1, 2, 3),
    (4, 8, 16, 32),
    (10,),
    (1,),  # Single element tensor
]


@pytest.mark.sym_size
@pytest.mark.parametrize("shape", SHAPES_FOR_SYMSIZE)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sym_size(shape, dtype):
    # sym_size is a metadata operation that returns a Python list, not a tensor,
    # so we use plain assert instead of gems_assert_close/gems_assert_equal.
    x = torch.randn(*shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    # Get sym_size result from reference (PyTorch default)
    ref_out = torch.ops.aten.sym_size(ref_x)
    # Get sym_size result from FlagGems
    with flag_gems.use_gems():
        act_out = torch.ops.aten.sym_size(x)

    # Compare the returned lists
    assert act_out == ref_out, f"sym_size mismatch: got {act_out}, expected {ref_out}"
