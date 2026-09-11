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
from . import conftest as cfg

if cfg.QUICK_MODE:
    SPECIAL_I0E_SHAPES = [(2, 3)]
else:
    SPECIAL_I0E_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.special_i0e
@pytest.mark.parametrize("shape", SPECIAL_I0E_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_i0e(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_i0e(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_i0e(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_i0e(x)
    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.special_i0e_out
@pytest.mark.parametrize("shape", SPECIAL_I0E_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_i0e_out(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        out_ref = torch.empty_like(ref_x, dtype=torch.float32)
        ref_out = torch.ops.aten.special_i0e.out(ref_x.float(), out=out_ref)
        out_ref = out_ref.to(dtype)
        ref_out = out_ref
    else:
        out_ref = torch.empty_like(ref_x)
        ref_out = torch.ops.aten.special_i0e.out(ref_x, out=out_ref)
    out_act = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_i0e.out(x, out=out_act)
    utils.gems_assert_close(act_out, ref_out, dtype)
    utils.gems_assert_close(out_act, out_ref, dtype)
