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
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

vendor_name = flag_gems.vendor_name


@pytest.mark.vdot
@pytest.mark.parametrize("M", utils.UT_SHAPES_1D)
@pytest.mark.parametrize(
    "is_conj", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.cfloat])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_vdot(M, is_conj, dtype, stride):
    if vendor_name in ["kunlunxin", "tsingmicro"]:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1_is_conj, inp2_is_conj = is_conj

    if vendor_name in ["mthreads", "tsingmicro"]:
        inp1 = torch.randn(M, dtype=dtype, device="cpu")
        inp2 = torch.randn(M, dtype=dtype, device="cpu")

    elif vendor_name == "ascend" and dtype == torch.cfloat:
        pytest.skip("Issue #2859: Skipping torch.cfloat tests on Ascend platform")

    elif vendor_name == "tsingmicro" and dtype == torch.cfloat:
        pytest.skip("Issue #2859: Skipping torch.cfloa tests on tsingmicro platform")

    elif vendor_name == "kunlunxin" and dtype == torch.cfloat:
        inp1 = torch.randn(M, dtype=dtype, device="cpu")
        inp2 = torch.randn(M, dtype=dtype, device="cpu")

    else:
        inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(M, dtype=dtype, device=flag_gems.device)

    inp1 = inp1[::stride]
    inp2 = inp2[::stride]

    if inp1_is_conj:
        inp1 = inp1.conj()
    if inp2_is_conj:
        inp2 = inp2.conj()

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    with flag_gems.use_gems():
        if flag_gems.vendor_name == "mthreads":
            res_out = torch.vdot(
                inp1.to(device=flag_gems.device), inp2.to(device=flag_gems.device)
            )
        else:
            res_out = torch.vdot(inp1, inp2)
    ref_out = torch.vdot(ref_inp1, ref_inp2)
    utils.gems_assert_close(res_out, ref_out, dtype)
