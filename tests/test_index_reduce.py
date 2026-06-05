import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.index_reduce
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", ["prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce(reduce, include_self, dtype):
    # Test with a simple case: 2D tensor, dim=0
    # (5, 3) shape chosen to cover cases with duplicate and non-duplicate indices
    shape = (5, 3)
    device = flag_gems.device

    # Create input tensor filled with initial values
    inp = torch.full(shape, 2.0, dtype=dtype, device=device)
    # Source tensor to accumulate - 4 rows to match index length
    source = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=dtype, device=device
    )
    # Index tensor - specifies which rows to accumulate to, includes duplicate index 0
    index = torch.tensor([0, 4, 2, 0], dtype=torch.long, device=device)

    ref_inp = utils.to_reference(inp.clone())
    ref_source = utils.to_reference(source)
    ref_index = utils.to_reference(index)

    ref_out = torch.index_reduce(
        ref_inp, 0, ref_index, ref_source, reduce, include_self=include_self
    )
    with flag_gems.use_gems():
        res_out = torch.index_reduce(
            inp, 0, index, source, reduce, include_self=include_self
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.index_reduce_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", ["prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
def test_index_reduce_(reduce, include_self, dtype):
    # Test in-place version
    # (5, 3) shape chosen to cover cases with duplicate and non-duplicate indices
    shape = (5, 3)
    device = flag_gems.device

    inp = torch.full(shape, 2.0, dtype=dtype, device=device)
    source = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=dtype, device=device
    )
    index = torch.tensor([0, 4, 2, 0], dtype=torch.long, device=device)

    ref_inp = utils.to_reference(inp.clone())
    ref_source = utils.to_reference(source)
    ref_index = utils.to_reference(index)

    ref_out = ref_inp.index_reduce_(
        0, ref_index, ref_source, reduce, include_self=include_self
    )
    with flag_gems.use_gems():
        res_out = inp.index_reduce_(0, index, source, reduce, include_self=include_self)

    utils.gems_assert_close(res_out, ref_out, dtype)
    # Verify the mutated input tensor is also correct (Rule 43)
    ref_inp_gems = utils.to_reference(inp)
    utils.gems_assert_close(ref_inp_gems, ref_out, dtype)
