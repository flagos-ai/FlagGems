import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)

device = flag_gems.device


@pytest.mark.unique2
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("sorted", [True])
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [False, True])
def test_unique2(shape, dtype, sorted, return_inverse, return_counts):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10, 10, shape, device=flag_gems.device).to(dtype)

    ref_inp = utils.to_reference(inp, False)

    if return_counts:
        if return_inverse:
            with flag_gems.use_gems():
                res_out, res_unique_order, res_counts = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_unique_order, ref_counts = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )

            assert res_out.numel() == ref_out.numel()

            utils.gems_assert_equal(res_unique_order, ref_unique_order)
        else:
            with flag_gems.use_gems():
                res_out, res_counts = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_counts = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )

            assert res_out.numel() == ref_out.numel()

        utils.gems_assert_equal(res_counts, ref_counts)
    else:
        if return_inverse:
            with flag_gems.use_gems():
                res_out, res_unique_order = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_unique_order = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )

            assert res_out.numel() == ref_out.numel()

            utils.gems_assert_equal(res_unique_order, ref_unique_order)
        else:
            with flag_gems.use_gems():
                res_out = torch.unique(
                    inp,
                    sorted=sorted,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out = torch.unique(
                ref_inp,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )
            assert res_out.numel() == ref_out.numel()

    utils.gems_assert_equal(res_out, ref_out)
