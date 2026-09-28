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

_SPECIAL_I1E_LOGGER = flag_gems.special_i1e.__module__
_SPECIAL_I1E_OUT_LOGGER = flag_gems.special_i1e_out.__module__


@pytest.mark.special_i1e
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_i1e(shape, dtype, caplog):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.i1e(ref_inp)
    with caplog.at_level("DEBUG", logger=_SPECIAL_I1E_LOGGER):
        with flag_gems.use_gems():
            res_out = torch.special.i1e(inp)
    assert f"{utils.gems_log_prefix(flag_gems.special_i1e)} SPECIAL_I1E" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_i1e_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_i1e_out(shape, dtype, caplog):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten.special_i1e.out(ref_inp, out=ref_out)

    out = torch.empty_like(inp)
    with caplog.at_level("DEBUG", logger=_SPECIAL_I1E_OUT_LOGGER):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_i1e.out(inp, out=out)

    expected_prefix = utils.gems_log_prefix(flag_gems.special_i1e_out)
    assert f"{expected_prefix} SPECIAL_I1E_OUT" in caplog.text
    assert res_out is out
    utils.gems_assert_close(res_out, ref_out, dtype)
