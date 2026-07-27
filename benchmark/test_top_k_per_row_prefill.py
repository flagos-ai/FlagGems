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

"""Benchmark for top_k_per_row_prefill (DeepSeek V4 sparse attention).

Shapes match DeepSeek V4 production config:
    - vocab_size=129280: DeepSeek V4 vocabulary size (full BPE vocab)
    - top_k=1024: number of KV cache slots selected per token in sparse attention
    - num_rows=1: single-token decode (latency-critical path)
    - num_rows=32: typical prefill micro-batch
    - num_rows=64: larger prefill batch
    - num_rows=2048: max prefill sequence length

The baseline uses vLLM's persistent_topk CUDA kernel when available,
falling back to torch.topk for the selection-only theoretical minimum.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import top_k_per_row_prefill

from . import base

device = flag_gems.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)

# --- vLLM CUDA baseline (preferred) with PyTorch fallback ---
try:
    import vllm._custom_ops  # noqa: F401 — loads torch.ops._C

    def _vllm_top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    ):
        torch.ops._C.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    _vllm_top_k_per_row_prefill = None


class TopKPerRowPrefillBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, vocab_size, top_k, stride0, stride1"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # DeepSeek-V4-Flash
            (4, 8193, 512, 8456, 1),
            (16383, 4095, 512, 4352, 1),
            (4, 16385, 512, 16648, 1),
            (12961, 4100, 512, 4360, 1),
            (16380, 5115, 512, 5376, 1),
            (4100, 1025, 512, 1288, 1),
        ]

    def get_input_iter(self, dtype):
        for num_rows, vocab_size, top_k, stride0, stride1 in self.shapes:
            torch.manual_seed(42)
            buf = torch.randn(
                (num_rows - 1) * stride0 + (vocab_size - 1) * stride1 + 1,
                device=device,
                dtype=torch.float32,
            )
            logits = torch.as_strided(buf, (num_rows, vocab_size), (stride0, stride1))
            row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
            row_ends = torch.full(
                (num_rows,), vocab_size, dtype=torch.int32, device=device
            )
            indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)

            yield logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k


@pytest.mark.top_k_per_row_prefill
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
def test_top_k_per_row_prefill():
    bench = TopKPerRowPrefillBenchmark(
        op_name="top_k_per_row_prefill",
        torch_op=_vllm_top_k_per_row_prefill,
        gems_op=top_k_per_row_prefill,
        dtypes=[torch.float32],
    )
    bench.run()
