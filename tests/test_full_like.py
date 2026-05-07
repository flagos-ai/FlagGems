import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


@pytest.mark.full_like
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
@pytest.mark.parametrize(
    "fill_value", [3.1415926, 2, False, float("inf"), float("nan")]
)
def test_full_like(shape, dtype, fill_value):
    if isinstance(fill_value, float) and (
        math.isinf(fill_value) or math.isnan(fill_value)
    ):
        if dtype not in utils.ALL_FLOAT_DTYPES:
            # Skipping inf/nan test for non-float dtypes
            return

    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    # without dtype
    ref_out = torch.full_like(ref_inp, fill_value)
    with flag_gems.use_gems():
        res_out = torch.full_like(inp, fill_value)
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)

    # with dtype
    ref_out = torch.full_like(ref_inp, fill_value, dtype=dtype)
    with flag_gems.use_gems():
        res_out = torch.full_like(inp, fill_value, dtype=dtype)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
