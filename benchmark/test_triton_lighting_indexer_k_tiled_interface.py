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
from flag_gems.deprecated.DSA.indexer_k_tiled import (
    triton_lighting_indexer_k_tiled_interface,
)
from tests.test_DSA.test_indexer_k_tiled import (
    make_lighting_indexer_input,
    reference_lighting_indexer_implementation,
)

from . import base

# (seq_len_q, seq_len_kv, num_heads, qk_dim)
LIGHTING_INDEXER_SHAPES = [
    (1024, 2048, 16, 64),
    (2048, 4096, 32, 64),
    (4096, 8192, 64, 128),
]


def reference_lighting_indexer_logits(q, kv, weights, ks, ke):
    # tests/ reference returns (logits, cost); the operator returns logits only.
    logits, _ = reference_lighting_indexer_implementation(q, kv, weights, ks, ke)
    return logits


class LightingIndexerBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LIGHTING_INDEXER_SHAPES
        self.shape_desc = "seq_len_q, seq_len_kv, num_heads, qk_dim"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for seq_len_q, seq_len_kv, num_heads, qk_dim in self.shapes:
            yield make_lighting_indexer_input(
                seq_len_q,
                seq_len_kv,
                num_heads,
                qk_dim,
                dtype,
                self.device,
            )


@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.triton_lighting_indexer_k_tiled_interface
def test_triton_lighting_indexer_k_tiled_interface():
    bench = LightingIndexerBenchmark(
        op_name="triton_lighting_indexer_k_tiled_interface",
        # Baseline reuses the torch reference from
        # tests/test_DSA/torch_src/fp8_lighting_indexer.py (per-head einsum,
        # relu and weighted reduction): the same semantic reference the
        # accuracy gate uses.
        torch_op=reference_lighting_indexer_logits,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(triton_lighting_indexer_k_tiled_interface)
    bench.run()
