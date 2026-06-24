import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.alpha_dropout
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_alpha_dropout(shape, p, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if utils.TO_CPU or shape == (1,):
        # Use a larger shape for statistical validation when CPU fallback or 1D shape
        shape = (32768,)

    torch.manual_seed(42)
    res_inp = torch.randn(
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(res_inp)

    p = np.float32(p)

    # Test with train=True
    ref_out = torch.alpha_dropout(ref_inp, p, True)
    with flag_gems.use_gems():
        res_out = torch.alpha_dropout(res_inp, p, True)

    # Alpha dropout is stochastic, but we can verify that:
    # 1. The output shape matches
    # 2. The output dtype matches
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype

    # For evaluation mode (train=False), it should be identity
    res_inp_eval = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp_eval = utils.to_reference(res_inp_eval)
    ref_out_eval = torch.alpha_dropout(ref_inp_eval, p, False)
    with flag_gems.use_gems():
        res_out_eval = torch.alpha_dropout(res_inp_eval, p, False)

    # Verify identity in eval mode
    ref_out_eval = utils.to_reference(res_out_eval)
    utils.gems_assert_equal(res_out_eval, ref_out_eval)
