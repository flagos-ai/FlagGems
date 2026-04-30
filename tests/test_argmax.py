import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    DIM_LIST = [1]
    FLOAT_DTYPES = [torch.float32]
else:
    DIM_LIST = [0, 1]
    FLOAT_DTYPES = utils.FLOAT_DTYPES

EMPTY_SHAPES = [(0, 5), (3, 0, 4), (2, 5, 0), (0,)]


# TODO: There are some bugs in argmax with large size.
@pytest.mark.argmax
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES + EMPTY_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST + [None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmax(shape, dim, keepdim, dtype):
    rank = len(shape)
    is_empty_tensor = any(d == 0 for d in shape)

    if dim is not None:
        if rank == 0 or dim >= rank or dim < -rank:
            pytest.skip(f"Dimension {dim} is out of bound for shape {shape}")

    if is_empty_tensor:
        if dim is None:
            pytest.skip(
                "PyTorch reference requires dim specification for empty tensor."
            )

        dim_index = dim % rank
        if shape[dim_index] == 0:
            pytest.skip(
                f"PyTorch reference prohibits reduction on zero-sized dimension ({dim})."
            )

    if is_empty_tensor:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)

    ref_out = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.argmax(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)
