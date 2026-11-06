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
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [1]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]


@pytest.mark.argmin
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST + [None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.INT_DTYPES)
def test_argmin(shape, dim, keepdim, dtype):
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-1024, 1024, size=shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.argmin(ref_inp, dim=dim, keepdim=keepdim)

    res_out = flag_gems.argmin(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.argmin
@pytest.mark.parametrize("shape", [(1,), (1, 1), (3, 5), (2, 3), (7, 4097)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_argmin_k1_tail_mask(shape, dtype):
    # Shapes whose M x N is smaller than the BLOCK_M x BLOCK_N tile used by
    # argmin_kernel_opt_k1; tail rows/cols must be neither read nor written.
    if dtype in utils.INT_DTYPES:
        inp = torch.randint(-1024, 1024, size=shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.argmin(utils.to_reference(inp), dim=-1)
    res_out = flag_gems.argmin(inp, dim=-1)

    utils.gems_assert_equal(res_out, ref_out)
