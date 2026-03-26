"""
Sparse MLA performance benchmark: FlagGems Triton kernel vs vLLM CUDA kernel.

Model configurations:
- GLM-5 CONFIG1: d_qk=512, h_q=64, topk=512
- GLM-5 CONFIG2: d_qk=512, h_q=128, topk=1024
- DeepSeek-V3.2: d_qk=576, h_q=128, topk=2048

Usage:
    pytest benchmark/test_sparse_mla_perf.py -v -s
"""

import pytest
import torch

import flag_gems

from .performance_utils import Benchmark

device = flag_gems.device


def is_flashmla_available():
    try:
        import vllm._flashmla_C  # noqa: F401

        return True
    except ImportError:
        return False


HAS_FLASHMLA = is_flashmla_available()


def _vllm_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v):
    """vLLM flash-mla CUDA sparse prefill forward (baseline)."""
    result = torch.ops._flashmla_C.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, None, None
    )
    return result[0], result[1], result[2]


def _gems_sparse_prefill_fwd(q, kv, indices, sm_scale, d_v):
    """FlagGems Triton sparse prefill forward."""
    from flag_gems.fused.DSA.sparse_mla import sparse_prefill_fwd

    return sparse_prefill_fwd(q, kv, indices, sm_scale, d_v)


class SparseMlaBenchmark(Benchmark):
    """Benchmark for sparse MLA: FlagGems Triton vs vLLM CUDA kernel."""

    def __init__(self):
        super().__init__(
            op_name="sparse_mla",
            torch_op=_vllm_sparse_prefill_fwd,
            dtypes=[torch.bfloat16],
        )
        self.set_gems(_gems_sparse_prefill_fwd)

    def set_shapes(self, shape_file_path=None):
        # (s_q, d_qk, h_q, topk, s_kv)
        # Must match test cases in tests/test_DSA/test_sparse_mla_ops.py
        model_configs = [
            # GLM-5 CONFIG1: d_qk=512, h_q=64, topk=512
            (512, 64, 512, 8192),
            (512, 64, 512, 32768),
            (512, 64, 512, 49152),
            (512, 64, 512, 65536),
            # GLM-5 CONFIG2: d_qk=512, h_q=128, topk=1024
            (512, 128, 1024, 8192),
            (512, 128, 1024, 32768),
            (512, 128, 1024, 49152),
            (512, 128, 1024, 65536),
            # DeepSeek-V3.2: d_qk=576, h_q=128, topk=2048
            (576, 128, 2048, 8192),
            (576, 128, 2048, 32768),
            (576, 128, 2048, 65536),
            (576, 128, 2048, 98304),
            (576, 128, 2048, 131072),
        ]
        self.shapes = []
        for s_q in [1, 4096]:
            for d_qk, h_q, topk, s_kv in model_configs:
                self.shapes.append((s_q, d_qk, h_q, topk, s_kv))

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            s_q, d_qk, h_q, topk, s_kv = shape
            h_kv = 1
            d_v = 512
            sm_scale = d_qk ** (-0.5)
            actual_topk = min(topk, s_kv)

            torch.manual_seed(42)
            q = torch.randn(s_q, h_q, d_qk, dtype=cur_dtype, device=device)
            kv = torch.randn(s_kv, h_kv, d_qk, dtype=cur_dtype, device=device)
            indices = torch.randint(
                0, s_kv, (s_q, h_kv, actual_topk), dtype=torch.int32, device=device
            )

            yield (q, kv, indices, sm_scale, d_v)


@pytest.mark.skipif(not HAS_FLASHMLA, reason="requires vLLM flash-mla CUDA kernel")
@pytest.mark.sparse_mla_benchmark
@pytest.mark.performance
def test_perf_sparse_mla():
    """Benchmark FlagGems sparse MLA vs vLLM flash-mla CUDA kernel."""
    bench = SparseMlaBenchmark()
    bench.run()
