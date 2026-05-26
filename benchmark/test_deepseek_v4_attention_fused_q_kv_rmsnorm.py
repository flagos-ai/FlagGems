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
        self.shapes = [(32, 64 * 576, 576)]

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
