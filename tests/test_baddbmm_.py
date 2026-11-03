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

from .accuracy_utils import FLOAT_DTYPES as ORIG_FLOAT_DTYPES
from .accuracy_utils import SCALARS, gems_assert_close, to_reference
from .conftest import QUICK_MODE

if QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = ORIG_FLOAT_DTYPES


def _reference_input(value):
    # PPU lacks native inplace BLAS, and Enflame native baddbmm_ can return
    # incorrect values. CPU ATen supplies an independent numerical oracle;
    # cloning keeps reference mutation separate from candidate inputs.
    if flag_gems.vendor_name in {"thead", "enflame"}:
        return value.cpu().double().clone()
    reference = to_reference(value, True)
    # This is an independent value reference, so its output need not share
    # the candidate's strided layout. muDNN requires a contiguous output.
    if flag_gems.vendor_name == "mthreads":
        return reference.clone(memory_format=torch.contiguous_format)
    return reference.clone()


@pytest.mark.baddbmm_
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_baddbmm_(M, N, K, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3794: not working")

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    inp = torch.randn((batch, M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = _reference_input(mat1)
    ref_mat2 = _reference_input(mat2)
    ref_inp = _reference_input(inp)

    alpha = beta = scalar

    ref_out = ref_inp.baddbmm_(ref_mat1, ref_mat2, alpha=alpha, beta=beta)

    inp1 = inp.clone()
    res_out = flag_gems.baddbmm_(inp1, mat1, mat2, alpha=alpha, beta=beta)

    assert res_out.device == inp.device and res_out.dtype == dtype
    gems_assert_close(res_out.cpu() if ref_out.device.type == "cpu" else res_out, ref_out, dtype, reduce_dim=K)
    gems_assert_close(inp1.cpu() if ref_inp.device.type == "cpu" else inp1, ref_inp, dtype, reduce_dim=K)
    assert res_out is inp1
