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

from flag_gems.fused.DSA.indexer_k_tiled import (
    triton_lighting_indexer_k_tiled_interface,
)

from . import base


def _lighting_ks_ke(q_len, kv_len, device):
    """Deterministic per-query [ks, ke) kv ranges (causal layout)."""
    ks = torch.zeros(q_len, dtype=torch.int32, device=device)
    ke = torch.minimum(
        torch.arange(1, q_len + 1, dtype=torch.int32, device=device),
        torch.full((q_len,), kv_len, dtype=torch.int32, device=device),
    )
    return ks, ke


def _lighting_indexer_input_fn(config, dtype, device):
    q_len, kv_len, num_heads, qk_dim = config
    q = torch.randn((q_len, num_heads, qk_dim), device=device, dtype=dtype)
    kv = torch.randn((kv_len, qk_dim), device=device, dtype=dtype)
    weights = torch.randn((q_len, num_heads), device=device, dtype=torch.float32)
    ks, ke = _lighting_ks_ke(q_len, kv_len, device)
    yield q, kv, weights, ks, ke


class LightingIndexerKTiledBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # (num_queries, num_kv, num_heads, qk_dim)
        self.shapes = [
            (256, 512, 16, 64),
            (512, 2048, 32, 64),
            (1024, 4096, 32, 128),
            (2048, 8192, 32, 128),
            (4096, 16384, 64, 128),
        ]
        self.shape_desc = "num_queries, num_kv, num_heads, qk_dim"

    def set_more_shapes(self):
        return []


@pytest.mark.triton_lighting_indexer_k_tiled_interface
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_triton_lighting_indexer_k_tiled_interface_benchmark():
    """
    Benchmark the FlagGems DSA fp8 lighting indexer K-tiled kernel.
    """
    bench = LightingIndexerKTiledBenchmark(
        op_name="triton_lighting_indexer_k_tiled_interface",
        torch_op=triton_lighting_indexer_k_tiled_interface,
        input_fn=_lighting_indexer_input_fn,
        dtypes=[torch.bfloat16],
    )
    bench.run()
