import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unique_consecutive
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("return_inverse", [True, False])
@pytest.mark.parametrize("return_counts", [False, True])
def test_accuracy_unique_consecutive(shape, dtype, return_inverse, return_counts):
    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        # Use integers with some consecutive duplicates
        inp = torch.randint(-5, 5, shape, device=flag_gems.device).to(dtype)

    ref_inp = utils.to_reference(inp, False)

    if return_counts:
        if return_inverse:
            with flag_gems.use_gems():
                res_out, res_inverse, res_counts = torch.unique_consecutive(
                    inp,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_inverse, ref_counts = torch.unique_consecutive(
                ref_inp,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )

            utils.gems_assert_equal(res_inverse, ref_inverse)

        else:
            with flag_gems.use_gems():
                res_out, res_counts = torch.unique_consecutive(
                    inp,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )

            ref_out, ref_counts = torch.unique_consecutive(
                ref_inp,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )

        utils.gems_assert_equal(res_counts, ref_counts)

    else:
        if return_inverse:
            with flag_gems.use_gems():
                res_out, res_inverse = torch.unique_consecutive(
                    inp,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out, ref_inverse = torch.unique_consecutive(
                ref_inp,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )

            utils.gems_assert_equal(res_inverse, ref_inverse)

        else:
            with flag_gems.use_gems():
                res_out = torch.unique_consecutive(
                    inp,
                    return_inverse=return_inverse,
                    return_counts=return_counts,
                )
            ref_out = torch.unique_consecutive(
                ref_inp,
                return_inverse=return_inverse,
                return_counts=return_counts,
            )

    utils.gems_assert_equal(res_out, ref_out)
