"""Benchmark for fused_recurrent_gated_delta_rule_packed_decode.

Compares the optimized FlagGems Triton kernel against the sglang reference
implementation across different batch sizes.
"""

import pytest
import torch

import flag_gems
from benchmark.base import Benchmark

try:
    from sglang.srt.layers.attention.fla.fused_recurrent import \
        fused_recurrent_gated_delta_rule_packed_decode as \
        base_fused_recurrent_gated_delta_rule_packed_decode

    SGLANG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    base_fused_recurrent_gated_delta_rule_packed_decode = None
    SGLANG_AVAILABLE = False


def _torch_op_wrapper(*args, **kwargs):
    """Wrapper: falls back to FlagGems op when sglang is unavailable."""
    if SGLANG_AVAILABLE:
        return base_fused_recurrent_gated_delta_rule_packed_decode(
            *args, **kwargs
        )
    return flag_gems.fused_recurrent_gated_delta_rule_packed_decode(
        *args, **kwargs
    )


class FusedRecurrentGatedDeltaRulePackedDecodeBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "B, H, HV, K, V"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default config matching DeepSeek production shape.
        self.H = 8
        self.HV = 16
        self.K = 128
        self.V = 128
        self.num_slots = 1024
        self.scale = 0.08838834764831845
        self.use_qk_l2norm = True

    def set_more_shapes(self):
        return [
            (1,),
            (4,),
            (8,),
            (16,),
            (32,),
            (64,),
            (128,),
            (256,),
        ]

    def get_input_iter(self, cur_dtype):
        for (B,) in self.shapes:
            yield self._build_inputs(B, cur_dtype)

    def _build_inputs(self, B: int, dtype: torch.dtype):
        device = flag_gems.device
        H, HV, K, V = self.H, self.HV, self.K, self.V

        # Packed mixed_qkv layout: [Q (H*K), K (H*K), V (HV*V)]
        qk_dim = 2 * H * K
        v_dim = HV * V
        mixed_qkv_dim = qk_dim + v_dim

        mixed_qkv = torch.randn((B, mixed_qkv_dim), device=device, dtype=dtype)
        a = torch.randn((B, HV), device=device, dtype=dtype)
        b = torch.randn((B, HV), device=device, dtype=dtype)
        A_log = torch.randn((HV,), device=device, dtype=dtype)
        dt_bias = torch.randn((HV,), device=device, dtype=dtype)
        initial_state = (
            torch.randn(
                (self.num_slots, HV, K, V), device=device, dtype=dtype
            )
            * 0.1
        )
        out = torch.empty((B, 1, HV, V), device=device, dtype=dtype)
        ssm_state_indices = torch.zeros(B, device=device, dtype=torch.long)

        return (
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            self.scale,
            initial_state,
            out,
            ssm_state_indices,
            self.use_qk_l2norm,
        )


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.skipif(
    not (SGLANG_AVAILABLE and torch.cuda.is_available()),
    reason="requires sglang installed and CUDA device",
)
def test_fused_recurrent_gated_delta_rule_packed_decode():
    bench = FusedRecurrentGatedDeltaRulePackedDecodeBenchmark(
        op_name="fused_recurrent_gated_delta_rule_packed_decode",
        torch_op=_torch_op_wrapper,
    )
    bench.set_gems(flag_gems.fused_recurrent_gated_delta_rule_packed_decode)
    bench.run()
