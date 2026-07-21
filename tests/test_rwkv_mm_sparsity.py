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


@pytest.mark.rwkv_mm_sparsity
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rwkv_mmsparsity(dtype):
    n = 16384
    embedding_dim = 4096

    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #3796: not working")

    k = torch.randn(n, dtype=dtype, device=flag_gems.device)
    k = torch.relu(k)
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(42)
        # kunlunxin sparsity test require 90% sparsity
        sparsity_levels = [0.9]
        for target_sparsity in sparsity_levels:
            threshold = torch.quantile(k.abs().to(torch.float32), target_sparsity).to(
                dtype
            )
            k = torch.relu(k - threshold)

    V_ = torch.randn(n, embedding_dim, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res = flag_gems.rwkv_mm_sparsity(k, V_)

    ref_k = utils.to_reference(k, True)
    ref_V_ = utils.to_reference(V_, True)
    ref_res = ref_k @ ref_V_

    utils.gems_assert_close(res, ref_res, dtype, equal_nan=True)
