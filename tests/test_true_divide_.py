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


@pytest.mark.true_divide_
@pytest.mark.true_divide_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_true_divide_tensor_inplace_dispatch(shape, dtype, caplog, request):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)
    torch.ops.aten.true_divide_.Tensor(ref_inp1, ref_inp2)
    inp1_ptr = inp1.data_ptr()
    with caplog.at_level("DEBUG", logger="flag_gems.ops.true_divide_"):
        res_out = flag_gems.true_divide_tensor_(inp1, inp2)
    registry = getattr(request.config, "_override_registry", None)
    if registry is None or registry.get_override("true_divide_tensor_") is None:
        assert (
            f"{utils.gems_log_prefix(flag_gems.true_divide_)} TRUE_DIVIDE_"
            in caplog.text
        )
    assert res_out is inp1
    assert inp1.data_ptr() == inp1_ptr
    utils.gems_assert_close(inp1, ref_inp1, dtype, equal_nan=True)
