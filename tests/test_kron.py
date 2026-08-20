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


@pytest.mark.kron
@pytest.mark.parametrize("shape", utils.KRON_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
)
def test_kron(shape, dtype):
    if dtype in utils.INT_DTYPES:
        inp1 = torch.randint(
            low=-10, high=10, size=shape[0], dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = torch.randint(
            low=-10, high=10, size=shape[1], dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    elif dtype in utils.FLOAT_DTYPES:
        inp1 = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape[1], dtype=dtype, device=flag_gems.device)
    else:
        inp1 = torch.randint(0, 2, size=shape[0], dtype=dtype, device=flag_gems.device)
        inp2 = torch.randint(0, 2, size=shape[1], dtype=dtype, device=flag_gems.device)

    # BUG: #2823 # Pytorch 2.0.1 Bfloat16 CPU Backend Precision fails
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.bfloat16:
        inp1 = torch.randn(shape[0], dtype=torch.float32, device=flag_gems.device)
        inp2 = torch.randn(shape[1], dtype=torch.float32, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.kron(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.kron(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
