import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unsafe_masked_index_put_accumulate
@pytest.mark.parametrize("shape", utils._UNSAFE_MASKED_INDEX_PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_masked_index_put_accumulate(shape, dtype):
    inp_shape, mask_shape, indices_shape, values_shape = shape
    assert (
        mask_shape == indices_shape == values_shape
    ), "mask, indices, and values must have same shape"

    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, mask_shape, dtype=torch.int32, device=flag_gems.device)
    indices = torch.randint(
        0, max(inp.numel(), 1), indices_shape, device=flag_gems.device
    )
    values = torch.randn(values_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone())
    ref_mask = utils.to_reference(mask.clone())
    ref_indices = utils.to_reference(indices.clone())
    ref_values = utils.to_reference(values.clone())

    op = torch._unsafe_masked_index_put_accumulate
    ref_out = op(ref_inp, ref_mask.clone(), (ref_indices.clone(),), ref_values)
    with flag_gems.use_gems():
        res_out = op(inp.clone(), mask, (indices,), values)

    utils.gems_assert_close(res_out, ref_out, dtype)
