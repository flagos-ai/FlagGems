import pytest
import torch

import flag_gems

from . import performance_utils as base


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


class FusedMoEINT8W8A16Benchmark(base.Benchmark):
    """
    Benchmark for fused_experts_impl with INT8 W8A16 weight-only quantization.

    Weights are pre-quantized to INT8 with per-channel scales.
    Activations remain in FP16/BF16 (no activation quantization).
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._int8_w8a16_input_fn(config, cur_dtype)

    def _int8_w8a16_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate INT8 weights one expert at a time to avoid OOM on large E.
        w1_int8 = torch.empty(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.int8,
        )
        w2_int8 = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.int8,
        )
        for e in range(num_experts):
            w1_int8[e] = to_int8(
                torch.randn(
                    intermediate_size * 2,
                    hidden_size,
                    device=device,
                    dtype=torch.float16,
                )
                * 50
            )
            w2_int8[e] = to_int8(
                torch.randn(
                    hidden_size, intermediate_size, device=device, dtype=torch.float16
                )
                * 50
            )

        # Per-channel scales [E, output_dim]
        w1_scale = (
            torch.rand(
                num_experts, intermediate_size * 2, device=device, dtype=torch.float32
            )
            * 0.01
            + 0.001
        )
        w2_scale = (
            torch.rand(num_experts, hidden_size, device=device, dtype=torch.float32)
            * 0.01
            + 0.001
        )

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (
            hidden_states,
            w1_int8,
            w2_int8,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        )


def _vllm_fused_moe_int8_w8a16_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Baseline: dequantize INT8 weights to bf16, then run FlagGems bf16
    fused_moe.  Measures the overhead of the dequant + bf16 path.

    NOTE: vLLM's INT8 W8A16 relies on specialised WNA16 kernels (CUDA or
    GPTQ/AWQ Triton) that are not directly comparable to the generic
    dequantize-then-GEMM approach, so we use a bf16 dequant baseline.
    """
    w1_deq = w1.to(hidden_states.dtype) * w1_scale.unsqueeze(-1).to(hidden_states.dtype)
    w2_deq = w2.to(hidden_states.dtype) * w2_scale.unsqueeze(-1).to(hidden_states.dtype)
    return flag_gems.fused_experts_impl(
        hidden_states.clone(),
        w1_deq,
        w2_deq,
        topk_weights,
        topk_ids,
    )


def _gems_fused_moe_int8_w8a16_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call FlagGems fused_experts_impl with INT8 W8A16."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_int8_w8a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_fused_moe_int8_w8a16():
    """
    Benchmark FlagGems fused_experts_impl with INT8 W8A16 quantization.

    Baseline is manual dequant + bf16 FlagGems (vLLM's INT8 W8A16 uses
    specialised WNA16 kernels not available via the generic Triton path).
    """
    bench = FusedMoEINT8W8A16Benchmark(
        op_name="fused_moe_int8_w8a16_gems_vs_bf16_deq",
        torch_op=_vllm_fused_moe_int8_w8a16_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_int8_w8a16_wrapper)
    bench.run()
