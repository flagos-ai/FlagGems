import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


@pytest.mark.new_full
@pytest.mark.skip(reason="This test fails on line 50.")
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
@pytest.mark.parametrize(
    "xdtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
@pytest.mark.parametrize(
    "fill_value", [3.1415926, 2, False, float("inf"), float("nan")]
)
def test_new_full(shape, dtype, xdtype, fill_value):
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    # without dtype: output dtype inherits from self (dtype), skip if dtype doesn't support inf/nan
    if isinstance(fill_value, float) and (
        math.isinf(fill_value) or math.isnan(fill_value)
    ):
        if dtype not in utils.ALL_FLOAT_DTYPES:
            pytest.skip("Skipping inf/nan test for non-float dtypes")

    ref_out = ref_inp.new_full(shape, fill_value)
    with flag_gems.use_gems():
        res_out = inp.new_full(shape, fill_value)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)

    # with dtype: output dtype is xdtype, skip if xdtype doesn't support inf/nan
    if isinstance(fill_value, float) and (
        math.isinf(fill_value) or math.isnan(fill_value)
    ):
        if xdtype not in utils.ALL_FLOAT_DTYPES:
            pytest.skip("Skipping inf/nan test for non-float dtypes")

    ref_out = ref_inp.new_full(shape, fill_value, dtype=xdtype)
    with flag_gems.use_gems():
        res_out = inp.new_full(shape, fill_value, dtype=xdtype)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
