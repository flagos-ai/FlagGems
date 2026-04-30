from math import ceil

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


DEFAULT_BLOCK_SHAPE = [128, 128]


class FusedMoEFP8BlockwiseBenchmark(base.Benchmark):
    """
    Benchmark for fused_experts_impl with FP8 W8A8 block-wise quantization.

    Weights are stored in FP8 E4M3 and accompanied by block scales.
    Activations are dynamically quantized per-token per-group inside the kernel.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.block_shape = DEFAULT_BLOCK_SHAPE

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
            # Qwen3.5-397B-A17B
            (1, 512, 4096, 1024, 10),
            (4, 512, 4096, 1024, 10),
            (16, 512, 4096, 1024, 10),
            (64, 512, 4096, 1024, 10),
            (128, 512, 4096, 1024, 10),
            (256, 512, 4096, 1024, 10),
        ]

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        for config in self.shapes:
            yield from self._fp8_blockwise_input_fn(config)

    def _fp8_blockwise_input_fn(self, config):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        block_n, block_k = self.block_shape
        device = flag_gems.device
        dtype = torch.bfloat16

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        w1_fp8 = (
            torch.randn(
                num_experts,
                intermediate_size * 2,
                hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            * (1.0 / hidden_size**0.5)
        ).to(torch.float8_e4m3fn)
        w2_fp8 = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                device=device,
                dtype=torch.bfloat16,
            )
            * (1.0 / intermediate_size**0.5)
        ).to(torch.float8_e4m3fn)

        w1_scale = (
            torch.rand(
                num_experts,
                ceil(intermediate_size * 2 / block_n),
                ceil(hidden_size / block_k),
                device=device,
                dtype=torch.float32,
            )
            + 0.01
        )
        w2_scale = (
            torch.rand(
                num_experts,
                ceil(hidden_size / block_n),
                ceil(intermediate_size / block_k),
                device=device,
                dtype=torch.float32,
            )
            + 0.01
        )

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(torch.float32)

        yield (
            hidden_states,
            w1_fp8,
            w2_fp8,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
        )


def _vllm_fused_moe_fp8_blockwise_wrapper(
    hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids
):
    """Wrapper to call vllm fused_experts_impl with block-wise FP8."""
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
        block_shape=DEFAULT_BLOCK_SHAPE,
    )


def _gems_fused_moe_fp8_blockwise_wrapper(
    hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids
):
    """Wrapper to call FlagGems fused_experts_impl with block-wise FP8."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=DEFAULT_BLOCK_SHAPE,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(
    not (HAS_VLLM_FUSED_MOE and CUDA_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture for FP8 blockwise",
)
def test_fused_moe_fp8_blockwise():
    """
    Benchmark FlagGems vs vLLM fused_experts_impl with FP8 W8A8 block-wise quantization.
    """
    bench = FusedMoEFP8BlockwiseBenchmark(
        op_name="fused_moe_fp8_blockwise_gems_vs_vllm",
        torch_op=_vllm_fused_moe_fp8_blockwise_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_fp8_blockwise_wrapper)
    bench.run()
