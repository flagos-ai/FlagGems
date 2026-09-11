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

from flag_gems.fused.deepseek_v4_attention_fused_q_kv_rmsnorm import fused_q_kv_rmsnorm

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        fused_q_kv_rmsnorm as vllm_fused_q_kv_rmsnorm,
    )

    _HAS_VLLM_FUSED_Q_KV_RMSNORM = True
except Exception:
    vllm_fused_q_kv_rmsnorm = None
    _HAS_VLLM_FUSED_Q_KV_RMSNORM = False

from . import base


class FusedQKVRMSNormBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "fused_q_kv_rmsnorm",
            vllm_fused_q_kv_rmsnorm,
            [torch.bfloat16],
            gems_op=fused_q_kv_rmsnorm,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [
            (1, 1536, 512),
            (32, 1536, 512),
            (128, 1536, 512),
            (512, 1536, 512),
            (2048, 1536, 512),
            (32, 64 * 576, 576),
            (128, 64 * 576, 576),
        ]

    def get_input_iter(self, dtype):
        for tokens, qdim, kvdim in self.shapes:
            qr = torch.randn((tokens, qdim), device="cuda", dtype=dtype)
            kv = torch.randn((tokens, kvdim), device="cuda", dtype=dtype)
            q_weight = torch.randn((qdim,), device="cuda", dtype=dtype)
            kv_weight = torch.randn((kvdim,), device="cuda", dtype=dtype)
            yield (qr, kv, q_weight, kv_weight, 1e-6)


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or (not _HAS_VLLM_FUSED_Q_KV_RMSNORM),
    reason="requires cuda and vllm deepseek_v4_ops.fused_q_kv_rmsnorm",
)
def test_fused_q_kv_rmsnorm_benchmark():
    FusedQKVRMSNormBenchmark().run()
