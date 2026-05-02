import pytest
import torch

import flag_gems

from . import base


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


class FusedMoEFP8Benchmark(base.Benchmark):
    """
    Benchmark for fused_experts_impl with FP8 W8A8 quantization.

    Weights are pre-quantized to FP8 E4M3 with per-expert scales.
    Activations are dynamically quantized per-tensor inside the kernel.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
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
            yield from self._fp8_input_fn(config, cur_dtype)

    def _fp8_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device
        fp8_dtype = torch.float8_e4m3fn

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate FP8 weights one expert at a time to avoid OOM on large E.
        w1_fp8 = torch.empty(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=fp8_dtype,
        )
        w2_fp8 = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=fp8_dtype,
        )
        for e in range(num_experts):
            w1_fp8[e] = to_fp8(
                torch.randn(
                    intermediate_size * 2,
                    hidden_size,
                    device=device,
                    dtype=torch.float16,
                )
            )
            w2_fp8[e] = to_fp8(
                torch.randn(
                    hidden_size, intermediate_size, device=device, dtype=torch.float16
                )
            )

        # Synthetic per-expert scales (representative of real quantization)
        w1_scale = (
            torch.rand(num_experts, device=device, dtype=torch.float32) * 0.01 + 0.001
        )
        w2_scale = (
            torch.rand(num_experts, device=device, dtype=torch.float32) * 0.01 + 0.001
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
            w1_fp8,
            w2_fp8,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        )


def _vllm_fused_moe_fp8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call vllm fused_experts_impl with FP8."""
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _gems_fused_moe_fp8_wrapper(
    hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale
):
    """Wrapper to call FlagGems fused_experts_impl with FP8."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


@pytest.mark.fused_moe_impl
@pytest.mark.skipif(
    not (HAS_VLLM_FUSED_MOE and CUDA_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture for FP8",
)
def test_fused_moe_fp8():
    """
    Benchmark FlagGems vs vLLM fused_experts_impl with FP8 W8A8 quantization.
    """
    bench = FusedMoEFP8Benchmark(
        op_name="fused_moe_fp8_gems_vs_vllm",
        torch_op=_vllm_fused_moe_fp8_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_fp8_wrapper)
    bench.run()
