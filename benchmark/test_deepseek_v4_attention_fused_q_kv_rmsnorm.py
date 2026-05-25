import pytest
import torch

from flag_gems.fused.deepseek_v4_attention_fused_q_kv_rmsnorm import fused_q_kv_rmsnorm

from . import base


def torch_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
    q = qr.float() * torch.rsqrt(
        torch.mean(qr.float() * qr.float(), dim=-1, keepdim=True) + eps
    )
    k = kv.float() * torch.rsqrt(
        torch.mean(kv.float() * kv.float(), dim=-1, keepdim=True) + eps
    )
    return (q * q_weight.float()).to(qr.dtype), (k * kv_weight.float()).to(kv.dtype)


class FusedQKVRMSNormBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "fused_q_kv_rmsnorm",
            torch_fused_q_kv_rmsnorm,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_fused_q_kv_rmsnorm_benchmark():
    FusedQKVRMSNormBenchmark().run()
