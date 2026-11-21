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

INDEX_COPY_BACKWARD_CASES = [
    ((1024, 64), 0),
    ((1, 2048, 8), 1),
    ((2, 8, 2048, 16), 2),
]


@pytest.mark.index_copy_backward
@pytest.mark.parametrize("shape, dim", INDEX_COPY_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_copy_backward(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    src = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    index = torch.randperm(shape[dim], device=flag_gems.device)

    res_out = flag_gems.index_copy(inp, dim, index, src)
    ref_inp = utils.to_reference(inp, True)
    ref_src = utils.to_reference(src, True)
    ref_index = utils.to_reference(index)

    ref_out = torch.index_copy(ref_inp, dim, ref_index, ref_src)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad, True)

    res_dinp, res_dsrc = torch.autograd.grad(res_out, (inp, src), out_grad)
    ref_dinp, ref_dsrc = torch.autograd.grad(ref_out, (ref_inp, ref_src), ref_out_grad)
    utils.gems_assert_close(res_dinp, ref_dinp, dtype)
    utils.gems_assert_close(res_dsrc, ref_dsrc, dtype)

    res_dinp_direct, res_dsrc_direct = flag_gems.index_copy_backward(
        out_grad, dim, index
    )
    utils.gems_assert_close(res_dinp_direct, ref_dinp, dtype)
    utils.gems_assert_close(res_dsrc_direct, ref_dsrc, dtype)
