import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.LayerNorm
@pytest.mark.parametrize(
    "shape",
    [
        (16, 32, 64),
        (8, 16, 32, 64),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_LayerNorm(shape, dtype):
    # LayerNorm normalizes over the last dimension
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    # Use flag_gems.LayerNorm which infers normalized_shape from last dimension
    ref_out = torch.nn.functional.layer_norm(ref_inp, (ref_inp.shape[-1],))
    with flag_gems.use_gems():
        res_out = flag_gems.LayerNorm(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
