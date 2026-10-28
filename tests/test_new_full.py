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
def test_new_full(shape, dtype, xdtype, fill_value):
    inp = torch.empty(size=shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    nonfinite_fill = isinstance(fill_value, float) and not math.isfinite(fill_value)
    for output_dtype in (None, xdtype):
        kwargs = {} if output_dtype is None else {"dtype": output_dtype}
        expected_dtype = dtype if output_dtype is None else output_dtype
        if nonfinite_fill and expected_dtype in utils.ALL_INT_DTYPES:
            # A nonfinite value cannot be represented by an integer output.
            # Check each variant independently so this does not hide a valid
            # explicit-dtype call in the same parameterized case.
            with pytest.raises(
                (RuntimeError, OverflowError), match="overflow|cannot be converted"
            ) as native_error:
                ref_inp.new_full(shape, fill_value, **kwargs)
            with pytest.raises(
                type(native_error.value), match="overflow|cannot be converted"
            ):
                flag_gems.new_full(inp, shape, fill_value, **kwargs)
            continue

        ref_out = ref_inp.new_full(shape, fill_value, **kwargs)
        res_out = flag_gems.new_full(inp, shape, fill_value, **kwargs)
        assert isinstance(res_out, torch.Tensor)
        assert res_out.device == inp.device
        assert res_out.dtype == expected_dtype
        assert res_out.shape == torch.Size(shape)
        assert res_out.data_ptr() != inp.data_ptr()
        utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
