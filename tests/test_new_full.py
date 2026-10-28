# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.new_full
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
@pytest.mark.parametrize("dtype_variant", ["inherit", "explicit"])
def test_new_full(shape, dtype, xdtype, fill_value, dtype_variant):
    kwargs = {} if dtype_variant == "inherit" else {"dtype": xdtype}
    expected_dtype = dtype if dtype_variant == "inherit" else xdtype
    nonfinite_fill = isinstance(fill_value, float) and not math.isfinite(fill_value)
    integer_nonfinite = nonfinite_fill and expected_dtype in utils.ALL_INT_DTYPES
    bool_nan = (
        isinstance(fill_value, float)
        and math.isnan(fill_value)
        and expected_dtype == torch.bool
    )
    if flag_gems.vendor_name == "ascend" and not cfg.TO_CPU:
        if integer_nonfinite:
            pytest.skip(
                "Ascend native new_full accepts nonfinite integer fills instead "
                "of the CPU overflow error"
            )
        if bool_nan:
            pytest.skip(
                "Ascend native new_full converts NaN to False for bool; CPU "
                "converts it to True"
            )

    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)
    if integer_nonfinite:
        with pytest.raises(
            (RuntimeError, OverflowError), match="overflow|cannot be converted"
        ) as native_error:
            ref_inp.new_full(shape, fill_value, **kwargs)
        with pytest.raises(
            type(native_error.value), match="overflow|cannot be converted"
        ):
            flag_gems.new_full(inp, shape, fill_value, **kwargs)
        return

    ref_out = ref_inp.new_full(shape, fill_value, **kwargs)
    res_out = flag_gems.new_full(inp, shape, fill_value, **kwargs)
    assert isinstance(res_out, torch.Tensor)
    assert res_out.device == inp.device
    assert res_out.dtype == expected_dtype
    assert res_out.shape == torch.Size(shape)
    assert res_out.data_ptr() != inp.data_ptr()
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
