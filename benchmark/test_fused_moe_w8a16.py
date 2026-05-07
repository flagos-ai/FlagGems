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


class FusedMoEMXQW8A16Benchmark(base.Benchmark):
    """
    Benchmark for flag_gems.fused_moe_mxq.fused_moe with W8A16 mixed precision.

    Uses QuantMode.W8A16: INT8 weights, FP16 activations.
    Tests SwiGLU MoE: y = W2 @ (silu(W1 @ x) * (W3 @ x))
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
            # DeepSeek-V3-like shapes (smaller E to avoid OOM)
            (1, 64, 7168, 2048, 8),
            (4, 64, 7168, 2048, 8),
            (16, 64, 7168, 2048, 8),
            (64, 64, 7168, 2048, 8),
            (128, 64, 7168, 2048, 8),
            (256, 64, 7168, 2048, 8),
            # Qwen3.5-397B-A17B (smaller E to avoid OOM)
            (1, 128, 4096, 1024, 10),
            (4, 128, 4096, 1024, 10),
            (16, 128, 4096, 1024, 10),
            (64, 128, 4096, 1024, 10),
            (128, 128, 4096, 1024, 10),
            (256, 128, 4096, 1024, 10),
        ]

    def set_more_metrics(self):
        # Display both QC TFLOPS (gems latency) and FP16 ref TFLOPS (torch latency_base).
        return ["tflops", "tflops_base"]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._w8a16_mxq_input_fn(config)

    def _w8a16_mxq_input_fn(self, config):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device
        dtype = torch.bfloat16

        from flag_gems.fused_moe_mxq import QuantConfig, QuantMode, quantize_weights_moe

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

        # Generate INT8 weights with scales (group-wise quantization)
        w1_fp16 = torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        ) * (1.0 / hidden_size**0.5)
        w2_fp16 = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        ) * (1.0 / intermediate_size**0.5)
        w3_fp16 = torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        ) * (1.0 / hidden_size**0.5)

        # Quantize to W8A16
        quant_config = QuantConfig(mode=QuantMode.W8A16, has_zero_point=False)
        w1_q, w1_scale, _ = quantize_weights_moe(w1_fp16, num_experts, quant_config)
        w2_q, w2_scale, _ = quantize_weights_moe(w2_fp16, num_experts, quant_config)
        w3_q, w3_scale, _ = quantize_weights_moe(w3_fp16, num_experts, quant_config)

        # Routing
        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (
            hidden_states,
            # FP16/BF16 reference weights (pure fused_experts_impl path)
            w1_fp16,
            w2_fp16,
            # Pre-quantized weights for QC W8A16 (fused_moe_mxq path)
            w1_q,
            w2_q,
            w3_q,
            w1_scale,
            w2_scale,
            w3_scale,
            topk_weights,
            topk_ids,
            num_experts,
            topk,
        )

    def get_tflops(
        self,
        op,
        hidden_states,
        w1_fp16,
        w2_fp16,
        w1_q,
        w2_q,
        w3_q,
        w1_scale,
        w2_scale,
        w3_scale,
        topk_weights,
        topk_ids,
        num_experts,
        topk,
    ):
        """
        Proxy FLOPs estimate for SwiGLU MoE.

        This is an algorithmic FLOPs estimate (not hardware-instruction FLOPs).
        It is derived strictly from tensor shapes to avoid hard-coded constants.

        For each (token, expert) dispatch, we approximate:
          - W1 projection: (H) x (Nw1)  => 2 * H * Nw1
          - W3 projection: (H) x (Nw1)  => 2 * H * Nw1
          - W2 projection: (I) x (H)    => 2 * H * I

        Total FLOPs:
          num_tokens * topk * (2*H*Nw1 + 2*H*Nw1 + 2*H*I)
        """
        # hidden_states: [num_tokens, H]
        num_tokens = int(hidden_states.shape[0])
        hidden_size = int(hidden_states.shape[1])
        # w1_fp16: [E, Nw1, H] where Nw1 is typically 2*I (gated)
        n_w1 = int(w1_fp16.shape[1])
        # w2_fp16: [E, H, I]
        intermediate_size = int(w2_fp16.shape[2])
        topk = int(topk)
        per_dispatch_flops = (
            2.0 * hidden_size * n_w1
            + 2.0 * hidden_size * n_w1
            + 2.0 * hidden_size * intermediate_size
        )
        total_flops = num_tokens * topk * per_dispatch_flops
        return total_flops


def _baseline_w8a16_mxq_wrapper(
    hidden_states,
    w1_fp16,
    w2_fp16,
    w1_q,
    w2_q,
    w3_q,
    w1_scale,
    w2_scale,
    w3_scale,
    topk_weights,
    topk_ids,
    num_experts,
    topk,
):
    """FP16/BF16 reference: run flag_gems.fused_experts_impl pure FP16 path."""
    del w1_q, w2_q, w3_q, w1_scale, w2_scale, w3_scale, num_experts, topk
    return flag_gems.fused_experts_impl(
        hidden_states.clone(),
        w1_fp16,
        w2_fp16,
        topk_weights,
        topk_ids,
    )


def _baseline_w8a16_mxq_wrapper_vllm(
    hidden_states,
    w1_fp16,
    w2_fp16,
    w1_q,
    w2_q,
    w3_q,
    w1_scale,
    w2_scale,
    w3_scale,
    topk_weights,
    topk_ids,
    num_experts,
    topk,
):
    """Wrapper to call vllm fused_experts_impl with W8A16 quantized weights."""
    del w1_fp16, w2_fp16, w3_q, w3_scale, num_experts, topk
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
        use_int8_w8a16=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _gems_fused_moe_mxq_w8a16_wrapper(
    hidden_states,
    w1_fp16,
    w2_fp16,
    w1_q,
    w2_q,
    w3_q,
    w1_scale,
    w2_scale,
    w3_scale,
    topk_weights,
    topk_ids,
    num_experts,
    topk,
):
    """Test flag_gems.fused_moe_mxq.fused_moe with W8A16."""
    del w1_fp16, w2_fp16
    from flag_gems.fused_moe_mxq import QuantConfig, QuantMode, fused_moe

    quant_config = QuantConfig(mode=QuantMode.W8A16, has_zero_point=False)
    return fused_moe(
        hidden_states,
        w1=None,
        w2=None,
        w3=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_config=quant_config,
        num_experts=num_experts,
        top_k=topk,
        w1_q=w1_q,
        w1_scales=w1_scale,
        w1_zeros=None,
        w2_q=w2_q,
        w2_scales=w2_scale,
        w2_zeros=None,
        w3_q=w3_q,
        w3_scales=w3_scale,
        w3_zeros=None,
    )


@pytest.mark.fused_moe
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_fused_moe_w8a16_mxq():
    """
    Benchmark flag_gems.fused_moe_mxq.fused_moe with W8A16 mixed precision.
    """
    bench = FusedMoEMXQW8A16Benchmark(
        op_name="fused_moe_w8a16_mxq_gems_vs_bf16_deq",
        torch_op=_baseline_w8a16_mxq_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_mxq_w8a16_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vLLM not installed")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_fused_moe_w8a16_mxq_gems_vs_vllm():
    """
    Benchmark flag_gems.fused_moe_mxq.fused_moe with W8A16 mixed precision.
    """
    bench = FusedMoEMXQW8A16Benchmark(
        op_name="fused_moe_w8a16_mxq_gems_vs_vllm",
        torch_op=_baseline_w8a16_mxq_wrapper_vllm,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_fused_moe_mxq_w8a16_wrapper)
    bench.run()
