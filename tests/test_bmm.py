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

import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
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
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm(monkeypatch, M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("#2799: Skipping fp32 bmm test on tsingmicro platform.")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm_non_contiguous(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2799: Skipping fp32 bmm test on tsingmicro.")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2_raw = torch.randn((batch, N, K), dtype=dtype, device=flag_gems.device)
    # make mat2 non-contiguous
    mat2 = mat2_raw.transpose(1, 2)

    if N > 1 and K > 1:
        assert not mat2.is_contiguous()
    else:
        # Skipping non-contiguous test for small N or K
        return

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.bmm_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm_out(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2799: Skipping fp32 bmm test on tsingmicro.")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((batch, M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        torch.bmm(mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)
