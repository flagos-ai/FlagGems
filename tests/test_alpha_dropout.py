import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.alpha_dropout
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_alpha_dropout(shape, p, dtype):
    utils.init_seed(0)

    if shape == (1,):
        shape = (32768,)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.alpha_dropout(inp, p, True)

    res_ref = utils.to_reference(res_out)
    inp_ref = utils.to_reference(inp)

    unique_vals = torch.unique(res_ref)
    assert len(unique_vals) > 1, "Output should not be all same value"

    kept = torch.isclose(res_ref, inp_ref, rtol=0.1, atol=0.1)
    keep_ratio = kept.float().mean().item()
    assert (
        abs(keep_ratio - (1 - p)) <= 0.1
    ), f"keep_ratio: {keep_ratio}, expected: {1 - p}"

    with flag_gems.use_gems():
        res_eval = torch.nn.functional.alpha_dropout(inp, p, False)
    ref_eval = inp.clone()
    utils.gems_assert_close(res_eval, ref_eval, dtype)
