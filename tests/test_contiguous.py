import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.contiguous
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_accuracy_contiguous(shape, dtype):
    if shape[0] <= 2:
        return

    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(
            low=-10000, high=10000, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)

    inp = inp[::2]
    assert inp.is_contiguous() is False

    ref_inp = utils.to_reference(inp)
    ref_out = ref_inp.contiguous()
    with flag_gems.use_gems():
        res_out = inp.contiguous()

    assert res_out.is_contiguous() is True
    assert res_out.is_contiguous() is True
    assert res_out.stride() == ref_out.stride()

    utils.gems_assert_equal(res_out, ref_out)
