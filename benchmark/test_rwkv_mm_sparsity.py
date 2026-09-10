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

from . import base, consts


# TODO(Qiming): Remove this class
class RWKVSparsityBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device):
    n = 16384
    embedding_dim = 4096

    V_ = torch.randn(n, embedding_dim, dtype=dtype, device=device)
    sparsity_levels = [0.9]
    for target_sparsity in sparsity_levels:
        k_sparse = torch.randn(n, dtype=dtype, device=device)
        threshold = torch.quantile(
            k_sparse.abs().to(torch.float32), target_sparsity
        ).to(dtype)
        k_sparse = torch.relu(k_sparse - threshold)

        yield k_sparse, V_


def torch_rwkv_mm_sparsity(k, v):
    return torch.mv(v.T, k)


@pytest.mark.rwkv_mm_sparsity
def test_rwkv_mm_sparsity():
    bench = RWKVSparsityBenchmark(
        input_fn=_input_fn,
        op_name="rwkv_mm_sparsity",
        torch_op=torch_rwkv_mm_sparsity,
        gems_op=flag_gems.rwkv_mm_sparsity,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
