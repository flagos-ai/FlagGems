import pytest
import torch

import flag_gems
from benchmark.performance_utils import Benchmark

try:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule as base_fused_recurrent_gated_delta_rule,
    )

    VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    base_fused_recurrent_gated_delta_rule = None
    VLLM_AVAILABLE = False


def rearrange_mixed_qkv(
    mixed_qkv, key_dim, value_dim, head_k_dim, head_v_dim, tp_size=1, contiguous=True
):
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, query.shape[0], -1, head_k_dim)
    key = key.view(1, key.shape[0], -1, head_k_dim)
    value = value.view(1, value.shape[0], -1, head_v_dim)
    if contiguous:
        return query.contiguous(), key.contiguous(), value.contiguous()
    else:
        return query, key, value


class FusedRecurrentGatedDeltaRuleBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]

    def __init__(self, *args, qkv_contiguous: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.qkv_contiguous = qkv_contiguous

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(128,), (512,)]
        self.shape_desc = "(T,)"

    def get_input_iter(self, cur_dtype):
        for (T,) in self.shapes:
            yield self._build_inputs(T, cur_dtype)

    def _build_inputs(self, T: int, dtype: torch.dtype):
        device = flag_gems.device
        B = 1
        H, HV, K, V = 16, 32, 128, 128
        tp_size = 4
        key_dim = H * K
        value_dim = HV * V

        assert key_dim % tp_size == 0 and value_dim % tp_size == 0

        mixed_qkv_dim = (2 * key_dim + value_dim) // tp_size
        total_tokens = B * T
        if self.qkv_contiguous:
            mixed_qkv = torch.randn(
                (total_tokens, mixed_qkv_dim), device=device, dtype=dtype
            )
        else:
            mixed_qkv_buffer = torch.randn(
                (total_tokens, mixed_qkv_dim, 2), device=device, dtype=dtype
            )
            mixed_qkv = mixed_qkv_buffer[:, :, 0]  # non-contiguous slice

        q, k, v = rearrange_mixed_qkv(
            mixed_qkv,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=K,
            head_v_dim=V,
            tp_size=tp_size,
            contiguous=self.qkv_contiguous,
        )

        HV_local = v.shape[2]
        g = torch.nn.functional.logsigmoid(
            torch.randn((B, T, HV_local), device=device, dtype=dtype)
        )
        beta = torch.rand(B, T, HV_local, device=device, dtype=dtype).sigmoid()
        cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.long)
        initial_state = torch.zeros((1024, HV_local, K, V), device=device, dtype=dtype)
        ssm_state_indices = torch.zeros(T, device=device, dtype=torch.long)
        scale = 0.08838834764831845

        # positional args follow fused_recurrent_gated_delta_rule_fwd signature
        return (
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            True,
            cu_seqlens,
            ssm_state_indices,
            None,
            True,
        )


def _torch_op_wrapper(*args, **kwargs):
    if VLLM_AVAILABLE:
        return base_fused_recurrent_gated_delta_rule(*args, **kwargs)
    return flag_gems.fused_recurrent_gated_delta_rule_fwd(*args, **kwargs)


@pytest.mark.skipif(flag_gems.device != "cuda", reason="benchmark requires CUDA device")
@pytest.mark.fused_recurrent_gated_delta_rule
@pytest.mark.parametrize("qkv_contiguous", [False])
def test_perf_fused_recurrent_gated_delta_rule(qkv_contiguous):
    bench = FusedRecurrentGatedDeltaRuleBenchmark(
        op_name=f"fused_recurrent_gated_delta_rule_{'contig' if qkv_contiguous else 'noncontig'}",
        torch_op=_torch_op_wrapper,
        qkv_contiguous=qkv_contiguous,
    )
    bench.set_gems(flag_gems.fused_recurrent_gated_delta_rule_fwd)
    bench.run()
