import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.isin
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("assume_unique", [False, True])
@pytest.mark.parametrize("invert", [False, True])
def test_accuracy_isin(shape, dtype, assume_unique, invert):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1 = torch.randint(-100, 100, shape, device=flag_gems.device).to(dtype)
    test_numel = inp1.numel() // 2 if inp1.numel() > 1 else 1
    test_shape = (test_numel,)
    inp2 = torch.randint(-10, 10, test_shape, device=flag_gems.device).to(dtype)
    inp1.ravel()[-1] = 0

    if assume_unique:
        inp1 = torch.unique(inp1.cpu()).to(flag_gems.device)
        inp2 = torch.unique(inp2.cpu()).to(flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    with flag_gems.use_gems():
        res_out = torch.isin(inp1, inp2, assume_unique=assume_unique, invert=invert)
    ref_out = torch.isin(ref_inp1, ref_inp2, assume_unique=assume_unique, invert=invert)

    utils.gems_assert_equal(res_out, ref_out)

    inp1_s = inp1.ravel()[0].item()
    with flag_gems.use_gems():
        res1_out = torch.isin(inp1_s, inp2, assume_unique=assume_unique, invert=invert)
    ref1_out = torch.isin(inp1_s, ref_inp2, assume_unique=assume_unique, invert=invert)

    utils.gems_assert_equal(res1_out, ref1_out)

    inp2_s = inp2.ravel()[0].item()
    with flag_gems.use_gems():
        res2_out = torch.isin(inp1, inp2_s, assume_unique=assume_unique, invert=invert)
    ref2_out = torch.isin(ref_inp1, inp2_s, assume_unique=assume_unique, invert=invert)

    utils.gems_assert_equal(res2_out, ref2_out)

    inp0 = torch.tensor([], device=flag_gems.device)
    ref_inp0 = utils.to_reference(inp0, False)
    with flag_gems.use_gems():
        res0_out = torch.isin(inp0, inp2, assume_unique=assume_unique, invert=invert)
    ref0_out = torch.isin(
        ref_inp0, ref_inp2, assume_unique=assume_unique, invert=invert
    )

    utils.gems_assert_equal(res0_out, ref0_out)
