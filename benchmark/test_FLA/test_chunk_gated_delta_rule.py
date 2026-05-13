# Copyright (c) 2025 FlagGems. All rights reserved.

import pytest
import torch
import triton

import flag_gems
import flag_gems.fused.FLA.utils as fla_utils
from benchmark.base import Benchmark


def _naive_recurrent_reference(q, k, v, beta, g, scale=None):
    """Naive token-by-token recurrent reference in pure PyTorch.

    Computes the same gated delta rule as the chunked operator, but one token
    at a time. Used as the baseline for performance comparison.
    All computation done in float32 for numerical stability, cast back at the end.
    """
    orig_dtype = q.dtype
    q, k, v, beta, g = map(lambda x: x.float(), [q, k, v, beta, g])
    if scale is None:
        scale = k.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]

    S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    o = torch.zeros(B, T, H, V, device=q.device, dtype=torch.float32)
    q_s = q * scale

    for i in range(T):
        k_i = k[:, i]
        q_i = q_s[:, i]
        v_i = v[:, i].clone()
        beta_i = beta[:, i]
        g_i = g[:, i]

        v_i = v_i - (S * k_i.unsqueeze(-1)).sum(-2)
        v_i = v_i * beta_i.unsqueeze(-1)
        S = S * g_i.exp().unsqueeze(-1).unsqueeze(-1)
        S = S + k_i.unsqueeze(-1) * v_i.unsqueeze(-2)
        o[:, i] = torch.einsum("bhd,bhdv->bhv", q_i, S)

    return o.to(orig_dtype)


class ChunkGatedDeltaRuleBenchmark(Benchmark):
    """Benchmark for chunk_gated_delta_rule operator.

    Compares the FlagGems chunked implementation against a naive PyTorch
    recurrent reference, measuring latency and speedup across different
    sequence lengths and data types.
    """

    DEFAULT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPE_DESC = "T"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_device_flags()

    def set_shapes(self, shape_file_path=None):
        """Override to use operator-specific shapes, ignoring the global YAML config.

        The global core_shapes.yaml contains 1D shapes like (1024*1024*1024,)
        which cause OOM when interpreted as sequence length for this 4D operator.
        """
        self.shapes = [
            (64,),
            (128,),
            (256,),
            (512,),
            (1024,),
        ]

    @staticmethod
    def _setup_device_flags():
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 10:  # Blackwell and newer
                fla_utils.is_tma_supported = False
                fla_utils.is_nvidia_hopper = False

        def _alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device=flag_gems.device)

        triton.set_allocator(_alloc_fn)

    def set_more_shapes(self):
        """Provide additional sequence lengths for comprehensive benchmarking."""
        return [
            (48,),
            (96,),
            (192,),
            (384,),
            (768,),
            (1536,),
            (2048,),
        ]

    def get_input_iter(self, cur_dtype):
        """Yield (q, k, v, beta, g, scale) for each shape."""
        for (T,) in self.shapes:
            yield self._build_inputs(T, cur_dtype)

    def _build_inputs(self, T: int, dtype: torch.dtype):
        device = flag_gems.device
        B, H, K, V = 1, 8, 64, 64

        q = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k = torch.nn.functional.normalize(k.float(), dim=-1, p=2).to(dtype)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype)
        beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
        g = torch.empty(B, T, H, device=device, dtype=dtype).uniform_(0.01, 0.03).log()

        scale = float(K**-0.5)
        return q, k, v, beta, g, scale


@pytest.mark.skipif(flag_gems.device != "cuda", reason="benchmark requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_perf_chunk_gated_delta_rule():
    bench = ChunkGatedDeltaRuleBenchmark(
        op_name="chunk_gated_delta_rule",
        torch_op=_naive_recurrent_reference,
    )
    bench.set_gems(lambda *a, **kw: flag_gems.chunk_gated_delta_rule(*a, **kw)[0])
    bench.run()
