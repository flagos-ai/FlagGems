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


@pytest.mark.logit
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logit(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.logit(inp, eps=1e-6)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logit_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logit_(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.logit_(eps=1e-6)
    with flag_gems.use_gems():
        res_out = inp.logit_(eps=1e-6)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logit_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logit_out(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.logit(ref_inp, eps=1e-6)

    out = torch.empty_like(inp)
    original_ptr = out.data_ptr()
    with flag_gems.use_gems():
        res_out = torch.logit(inp, eps=1e-6, out=out)

    assert res_out.data_ptr() == original_ptr
    assert out.data_ptr() == original_ptr
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.logit_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_logit_out_non_contiguous_out(dtype):
    torch.manual_seed(0)
    shape = (4, 7)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.logit(ref_inp, eps=1e-6)

    out_base = torch.empty(
        (shape[0], shape[1] * 2), device=flag_gems.device, dtype=dtype
    )
    out = out_base[:, ::2]
    assert not out.is_contiguous()

    with flag_gems.use_gems():
        res_out = torch.logit(inp, eps=1e-6, out=out)

    assert res_out.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref_out, dtype)
