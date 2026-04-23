import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


def replace_zeros(inp):
    return torch.where(inp == 0, 1, inp)


@pytest.mark.remainder
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)

    if cfg.TO_CPU:
        inp1 = replace_zeros(inp1)
        inp2 = replace_zeros(inp2)

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1 % ref_inp2
    with flag_gems.use_gems():
        res_out = inp1 % inp2

    utils.gems_assert_equal(res_out, ref_out)

    for d in inp2.flatten()[:2]:
        ref_d = utils.to_reference(d, False)
        ref_out = ref_inp1 % ref_d
        with flag_gems.use_gems():
            res_out = inp1 % d
        utils.gems_assert_equal(res_out, ref_out)

        ref_out = ref_d % ref_inp1
        with flag_gems.use_gems():
            res_out = d % inp1
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder_(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    if cfg.TO_CPU:
        inp1 = replace_zeros(inp1.clone())
        inp2 = replace_zeros(inp2)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.remainder_(ref_inp2)

    with flag_gems.use_gems():
        res_out = inp1.remainder_(inp2)

    utils.gems_assert_equal(res_out, ref_out)

    ref_inp1 = utils.to_reference(inp1.clone(), False)
    for d in inp2.flatten()[:2]:
        ref_d = utils.to_reference(d, False)
        ref_out = ref_inp1.remainder_(ref_d)

        with flag_gems.use_gems():
            res_out = inp1.remainder_(d)
        utils.gems_assert_equal(res_out, ref_out)
