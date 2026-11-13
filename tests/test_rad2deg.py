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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import gems_assert_close, to_reference

FLOAT_DTYPES = utils.FLOAT_DTYPES
POINTWISE_SHAPES = utils.POINTWISE_SHAPES

# rad2deg promotes integral inputs to float in ATen, so 1-byte ints are accepted.
RAD2DEG_DTYPES = FLOAT_DTYPES + utils.BYTE_INT_DTYPES


def _make_rad2deg_input(shape, dtype):
    if dtype in utils.BYTE_INT_DTYPES:
        return torch.randint(-100, 100, shape, dtype=torch.int32, device="cpu").to(
            dtype=dtype, device=flag_gems.device
        )
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


@pytest.mark.rad2deg
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", RAD2DEG_DTYPES)
def test_rad2deg(shape, dtype):
    inp = _make_rad2deg_input(shape, dtype)
    ref_inp = to_reference(inp)

    ref_out = torch.rad2deg(ref_inp)
    res_out = flag_gems.rad2deg(inp)

    gems_assert_close(res_out, ref_out, res_out.dtype)


@pytest.mark.rad2deg_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rad2deg_(shape, dtype):
    inp = _make_rad2deg_input(shape, dtype)
    ref_inp = to_reference(inp.clone())

    ref_out = ref_inp.rad2deg_()
    res_out = flag_gems.rad2deg_(inp)

    gems_assert_close(res_out, ref_out, dtype)
