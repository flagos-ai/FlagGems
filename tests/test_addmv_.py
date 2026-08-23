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
    MN_SHAPES = [
        (1, 32),
    ]
else:
    MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (5333, 497),
    ]


@pytest.mark.addmv_
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmv_(M, N, scalar, dtype):
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    inp = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_vec = utils.to_reference(vec, True)
    ref_inp = utils.to_reference(inp, True)

    alpha = beta = scalar

    ref_out = ref_inp.addmv_(ref_mat, ref_vec, alpha=alpha, beta=beta)

    inp1 = inp.clone()
    with flag_gems.use_gems():
        res_out = inp1.addmv_(mat, vec, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=N)
    utils.gems_assert_close(inp1, ref_inp, dtype, reduce_dim=N)
    assert res_out is inp1
