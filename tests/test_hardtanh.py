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


@pytest.mark.hardtanh
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    min_val, max_val = -1.0, 1.0
    ref_out = torch.nn.functional.hardtanh(ref_inp, min_val=min_val, max_val=max_val)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.hardtanh(
            res_inp, min_val=min_val, max_val=max_val
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardtanh_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh_out(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    min_val, max_val = -2.0, 2.0
    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.hardtanh.out(ref_inp, min_val=min_val, max_val=max_val, out=ref_out)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.hardtanh.out(res_inp, min_val=min_val, max_val=max_val, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.hardtanh_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardtanh_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    min_val, max_val = -1.0, 1.0
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(res_grad, True)

    ref_out = torch.ops.aten.hardtanh_backward(
        ref_grad, ref_inp, min_val=min_val, max_val=max_val
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardtanh_backward(
            res_grad, res_inp, min_val=min_val, max_val=max_val
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
