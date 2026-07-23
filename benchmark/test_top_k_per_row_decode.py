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

"""Benchmark for top_k_per_row_decode (DeepSeek V4 decode-phase top-K).

Shapes match DeepSeek V4 production config (vocab=129280, top_k=1024).
The baseline uses vLLM's CUDA kernel when available,
falling back to a pure-PyTorch reference (torch.topk).
"""

import inspect

import pytest
import torch
import triton.language as tl

from flag_gems.fused import top_k_per_row_decode

from . import base


def _has_histogram_mask():
    if not hasattr(tl, "histogram"):
        return False
    try:
        return "mask" in inspect.signature(tl.histogram).parameters
    except (ValueError, TypeError):
        return False


pytestmark = pytest.mark.skipif(
    not _has_histogram_mask(),
    reason="tl.histogram with mask parameter not available",
)

# --- vLLM CUDA baseline (preferred) with PyTorch fallback ---
try:
    import vllm._custom_ops  # noqa: F401 — loads torch.ops._C

    def _vllm_top_k_per_row_decode(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
    ):
        torch.ops._C.top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    _vllm_top_k_per_row_decode = None


class TopKPerRowDecodeBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, vocab_size, next_n, top_k, stride0, stride1"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # DeepSeek-V4-Flash
            (1, 262144, 1, 512, 262144, 1),
            (496, 262144, 1, 512, 262144, 1),
            (512, 262144, 1, 512, 262144, 1),
            (16, 262144, 1, 512, 262144, 1),
            (32, 262144, 1, 512, 262144, 1),
            (48, 262144, 1, 512, 262144, 1),
            (40, 262144, 1, 512, 262144, 1),
            (56, 262144, 1, 512, 262144, 1),
            (4, 262144, 1, 512, 262144, 1),
            (8, 262144, 1, 512, 262144, 1),
            (24, 262144, 1, 512, 262144, 1),
        ]

    def get_input_iter(self, dtype):
        for num_rows, vocab_size, next_n, top_k, stride0, stride1 in self.shapes:
            torch.manual_seed(42)
            buf = torch.randn(
                (num_rows - 1) * stride0 + (vocab_size - 1) * stride1 + 1,
                device=self.device,
                dtype=torch.float32,
            )
            logits = torch.as_strided(buf, (num_rows, vocab_size), (stride0, stride1))

            batch_size = num_rows // next_n
            seq_lens = torch.full(
                (batch_size,), vocab_size, dtype=torch.int32, device=self.device
            )
            indices = torch.zeros(
                (num_rows, top_k), dtype=torch.int32, device=self.device
            )

            yield (
                logits,
                next_n,
                seq_lens,
                indices,
                num_rows,
                stride0,
                stride1,
                top_k,
            )


@pytest.mark.top_k_per_row_decode
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
def test_top_k_per_row_decode():
    bench = TopKPerRowDecodeBenchmark(
        op_name="top_k_per_row_decode",
        torch_op=_vllm_top_k_per_row_decode,
        gems_op=top_k_per_row_decode,
        dtypes=[torch.float32],
    )
    bench.run()
