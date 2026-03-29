"""
Fused MoE performance benchmarks.

Benchmarks FlagGems fused_experts_impl against:
  - vLLM fused_experts_impl (bf16, fp8 per-tensor, fp8 block-wise, int8 w8a8)
  - bf16 dequant baselines for int8 w8a16 and int4 w4a16
  - hpc-ops block-wise fp8
  - SonicMoE bf16
"""

from math import ceil

import pytest
import torch

import flag_gems
from benchmark.performance_utils import Benchmark


MOE_SHAPES = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    # Mixtral-like
    (1, 8, 4096, 14336, 2),
    (4, 8, 4096, 14336, 2),
    (16, 8, 4096, 14336, 2),
    (64, 8, 4096, 14336, 2),
    (128, 8, 4096, 14336, 2),
    (256, 8, 4096, 14336, 2),
    (512, 8, 4096, 14336, 2),
    # DeepSeek-V3-like (TP=8 shard)
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

SONICMOE_SHAPES = [
    (1, 8, 4096, 14336, 2),
    (4, 8, 4096, 14336, 2),
    (16, 8, 4096, 14336, 2),
    (64, 8, 4096, 14336, 2),
    (128, 8, 4096, 14336, 2),
    (256, 8, 4096, 14336, 2),
    (512, 8, 4096, 14336, 2),
    (1, 256, 7168, 2048, 8),
    (4, 256, 7168, 2048, 8),
    (16, 256, 7168, 2048, 8),
    (64, 256, 7168, 2048, 8),
    (1, 512, 4096, 1024, 10),
    (4, 512, 4096, 1024, 10),
    (16, 512, 4096, 1024, 10),
    (64, 512, 4096, 1024, 10),
]

DEFAULT_BLOCK_SHAPE = [128, 128]


try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import moe_general_routing_inputs

    HAS_SONICMOE = True
except ImportError:
    HAS_SONICMOE = False

try:
    import hpc

    HAS_HPC = True
except ImportError:
    HAS_HPC = False


def is_hopper_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


HOPPER_AVAILABLE = is_hopper_available()


def _generate_bf16_inputs(config, dtype, device):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)
    return hidden_states, w1, w2, topk_weights, topk_ids


def _generate_fp8_per_tensor_inputs(config, dtype, device):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    fp8_info = torch.finfo(torch.float8_e4m3fn)

    w1 = torch.empty(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    w2 = torch.empty(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    for expert_idx in range(num_experts):
        w1[expert_idx] = torch.round(
            torch.randn(
                intermediate_size * 2,
                hidden_size,
                device=device,
                dtype=torch.float16,
            ).clamp(min=fp8_info.min, max=fp8_info.max)
        ).to(torch.float8_e4m3fn)
        w2[expert_idx] = torch.round(
            torch.randn(
                hidden_size,
                intermediate_size,
                device=device,
                dtype=torch.float16,
            ).clamp(min=fp8_info.min, max=fp8_info.max)
        ).to(torch.float8_e4m3fn)

    w1_scale = torch.rand(num_experts, device=device, dtype=torch.float32) * 0.01 + 0.001
    w2_scale = torch.rand(num_experts, device=device, dtype=torch.float32) * 0.01 + 0.001
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)
    return hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale


def _generate_fp8_blockwise_inputs(config, block_shape, device, sort_topk_ids=False):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    block_n, block_k = block_shape
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * (1.0 / hidden_size**0.5)
    ).to(torch.float8_e4m3fn)
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * (1.0 / intermediate_size**0.5)
    ).to(torch.float8_e4m3fn)
    w1_scale = torch.rand(
        num_experts,
        ceil(intermediate_size * 2 / block_n),
        ceil(hidden_size / block_k),
        device=device,
        dtype=torch.float32,
    ) + 0.01
    w2_scale = torch.rand(
        num_experts,
        ceil(hidden_size / block_n),
        ceil(intermediate_size / block_k),
        device=device,
        dtype=torch.float32,
    ) + 0.01

    if sort_topk_ids:
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device
        )
        topk_ids, _ = torch.sort(topk_ids, dim=1)
        topk_weights = (
            torch.randn((num_tokens, topk), dtype=torch.float32, device=device) / topk
        )
    else:
        gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(gating, dim=-1), topk, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(torch.float32)

    return hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids


def _generate_int8_inputs(config, dtype, device):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    w1 = torch.empty(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=torch.int8
    )
    w2 = torch.empty(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.int8
    )
    for expert_idx in range(num_experts):
        w1_fp32 = torch.randn(
            intermediate_size * 2, hidden_size, device=device, dtype=torch.float16
        ) * 50
        w2_fp32 = torch.randn(
            hidden_size, intermediate_size, device=device, dtype=torch.float16
        ) * 50
        w1[expert_idx] = torch.round(w1_fp32.clamp(min=-128, max=127)).to(torch.int8)
        w2[expert_idx] = torch.round(w2_fp32.clamp(min=-128, max=127)).to(torch.int8)

    w1_scale = (
        torch.rand(
            num_experts, intermediate_size * 2, device=device, dtype=torch.float32
        )
        * 0.01
        + 0.001
    )
    w2_scale = (
        torch.rand(num_experts, hidden_size, device=device, dtype=torch.float32) * 0.01
        + 0.001
    )
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)
    return hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale


def _generate_int4_inputs(config, dtype, device):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    w1 = torch.empty(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=torch.int8
    )
    w2 = torch.empty(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.int8
    )
    for expert_idx in range(num_experts):
        w1[expert_idx] = torch.randint(
            -8,
            8,
            (intermediate_size * 2, hidden_size),
            device=device,
            dtype=torch.int8,
        )
        w2[expert_idx] = torch.randint(
            -8,
            8,
            (hidden_size, intermediate_size),
            device=device,
            dtype=torch.int8,
        )

    w1_scale = (
        torch.rand(
            num_experts, intermediate_size * 2, device=device, dtype=torch.float32
        )
        * 0.01
        + 0.001
    )
    w2_scale = (
        torch.rand(num_experts, hidden_size, device=device, dtype=torch.float32) * 0.01
        + 0.001
    )
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)
    return hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")
def test_perf_fused_moe_bf16_gems_vs_vllm():
    class FusedMoEBf16Benchmark(Benchmark):
        def __init__(self, op_name, torch_op, dtypes):
            super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

        def set_shapes(self, shape_file_path=None):
            self.shapes = MOE_SHAPES

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield _generate_bf16_inputs(config, cur_dtype, flag_gems.device)

    def vllm_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
        return vllm_fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
        )

    def gems_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
        return flag_gems.fused_experts_impl(
            hidden_states, w1, w2, topk_weights, topk_ids
        )

    bench = FusedMoEBf16Benchmark(
        op_name="fused_moe_bf16_gems_vs_vllm",
        torch_op=vllm_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")
def test_perf_fused_moe_fp8_per_tensor_gems_vs_vllm():
    class FusedMoEFp8PerTensorBenchmark(Benchmark):
        def __init__(self, op_name, torch_op, dtypes):
            super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

        def set_shapes(self, shape_file_path=None):
            self.shapes = MOE_SHAPES

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield _generate_fp8_per_tensor_inputs(
                    config, cur_dtype, flag_gems.device
                )

    def vllm_wrapper(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
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

    def gems_wrapper(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
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

    bench = FusedMoEFp8PerTensorBenchmark(
        op_name="fused_moe_fp8_per_tensor_gems_vs_vllm",
        torch_op=vllm_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")
def test_perf_fused_moe_fp8_blockwise_gems_vs_vllm():
    class FusedMoEFp8BlockwiseVLLMBenchmark(Benchmark):
        def __init__(self, op_name, torch_op, dtypes):
            super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
            self.block_shape = DEFAULT_BLOCK_SHAPE

        def set_shapes(self, shape_file_path=None):
            self.shapes = MOE_SHAPES

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                torch.cuda.empty_cache()
                yield _generate_fp8_blockwise_inputs(
                    config, self.block_shape, flag_gems.device
                )

    def vllm_wrapper(hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids):
        return vllm_fused_experts_impl(
            hidden_states,
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

    def gems_wrapper(hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids):
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

    bench = FusedMoEFp8BlockwiseVLLMBenchmark(
        op_name="fused_moe_fp8_blockwise_gems_vs_vllm",
        torch_op=vllm_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_HPC, reason="hpc-ops not installed")
def test_perf_fused_moe_fp8_blockwise_gems_vs_hpc():
    class FusedMoEFp8BlockwiseHPCBenchmark(Benchmark):
        def __init__(self, op_name, torch_op, dtypes):
            super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
            self.block_shape = DEFAULT_BLOCK_SHAPE

        def set_shapes(self, shape_file_path=None):
            self.shapes = MOE_SHAPES

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                torch.cuda.empty_cache()
                num_experts = config[1]
                yield (
                    *_generate_fp8_blockwise_inputs(
                        config,
                        self.block_shape,
                        flag_gems.device,
                        sort_topk_ids=True,
                    ),
                    num_experts,
                )

    def hpc_wrapper(hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, num_experts):
        from flag_gems.ops.per_token_group_quant_fp8 import per_token_group_quant_fp8

        hidden_states_q, a1_scale = per_token_group_quant_fp8(
            hidden_states,
            group_size=DEFAULT_BLOCK_SHAPE[1],
            dtype=torch.float8_e4m3fn,
            column_major_scales=False,
            scale_ue8m0=False,
        )
        return hpc.fuse_moe_blockwise_fp8(
            hidden_states_q,
            a1_scale,
            w1,
            w1_scale,
            w2,
            w2_scale,
            topk_ids,
            topk_weights,
            0,
            num_experts,
        )

    def gems_wrapper(hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, num_experts):
        return flag_gems.fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=num_experts,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=DEFAULT_BLOCK_SHAPE,
        )

    bench = FusedMoEFp8BlockwiseHPCBenchmark(
        op_name="fused_moe_fp8_blockwise_gems_vs_hpc",
        torch_op=hpc_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_VLLM, reason="vllm not installed")
def test_perf_fused_moe_int8_gems_vs_vllm():
    class FusedMoEInt8Benchmark(Benchmark):
        def __init__(self, op_name, torch_op, dtypes):
            super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

        def set_shapes(self, shape_file_path=None):
            self.shapes = MOE_SHAPES

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield _generate_int8_inputs(config, cur_dtype, flag_gems.device)

    def vllm_wrapper(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
        return vllm_fused_experts_impl(
            hidden_states.clone(),
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation="silu",
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )

    def gems_wrapper(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
        return flag_gems.fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )

    bench = FusedMoEInt8Benchmark(
        op_name="fused_moe_int8_gems_vs_vllm",
        torch_op=vllm_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_perf_fused_moe_int8_w8a16_gems_vs_bf16_deq():
    class FusedMoEInt8W8A16Benchmark(Benchmark):
        def __init__(self, op_name, torch_op, dtypes):
            super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

        def set_shapes(self, shape_file_path=None):
            self.shapes = MOE_SHAPES

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield _generate_int8_inputs(config, cur_dtype, flag_gems.device)

    def dequant_bf16_wrapper(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
        w1_deq = w1.to(hidden_states.dtype) * w1_scale.unsqueeze(-1).to(hidden_states.dtype)
        w2_deq = w2.to(hidden_states.dtype) * w2_scale.unsqueeze(-1).to(hidden_states.dtype)
        return flag_gems.fused_experts_impl(
            hidden_states.clone(),
            w1_deq,
            w2_deq,
            topk_weights,
            topk_ids,
        )

    def gems_wrapper(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
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

    bench = FusedMoEInt8W8A16Benchmark(
        op_name="fused_moe_int8_w8a16_gems_vs_bf16_deq",
        torch_op=dequant_bf16_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(not HOPPER_AVAILABLE, reason="requires NVIDIA Hopper architecture")
def test_perf_fused_moe_int4_w4a16_gems_vs_bf16_deq():
    class FusedMoEInt4W4A16Benchmark(Benchmark):
        def __init__(self, op_name, torch_op, dtypes):
            super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

        def set_shapes(self, shape_file_path=None):
            self.shapes = MOE_SHAPES

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield _generate_int4_inputs(config, cur_dtype, flag_gems.device)

    def dequant_bf16_wrapper(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
        w1_deq = w1.to(hidden_states.dtype) * w1_scale.unsqueeze(-1).to(hidden_states.dtype)
        w2_deq = w2.to(hidden_states.dtype) * w2_scale.unsqueeze(-1).to(hidden_states.dtype)
        return flag_gems.fused_experts_impl(
            hidden_states.clone(),
            w1_deq,
            w2_deq,
            topk_weights,
            topk_ids,
        )

    def gems_wrapper(hidden_states, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale):
        return flag_gems.fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            use_int4_w4a16=True,
            per_channel_quant=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )

    bench = FusedMoEInt4W4A16Benchmark(
        op_name="fused_moe_int4_w4a16_gems_vs_bf16_deq",
        torch_op=dequant_bf16_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_wrapper)
    bench.run()


@pytest.mark.fused_moe
@pytest.mark.skipif(not HAS_SONICMOE, reason="sonicmoe not installed")
def test_perf_fused_moe_bf16_gems_vs_sonicmoe():
    class FusedMoESonicmoeBenchmark(Benchmark):
        def __init__(self, op_name, torch_op, dtypes):
            super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

        def set_shapes(self, shape_file_path=None):
            self.shapes = SONICMOE_SHAPES

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                num_tokens, _, _, intermediate_size, topk = config
                hidden_states, w1, w2, topk_weights, topk_ids = _generate_bf16_inputs(
                    config, cur_dtype, flag_gems.device
                )
                token_indices = (
                    torch.arange(num_tokens, dtype=torch.int32, device=flag_gems.device)
                    .unsqueeze(1)
                    .expand(-1, topk)
                    .reshape(-1)
                )
                expert_indices = topk_ids.reshape(-1)
                router_scores = topk_weights.reshape(-1)
                w1_sonic = torch.empty_like(w1)
                w1_sonic[:, 0::2, :] = w1[:, :intermediate_size, :]
                w1_sonic[:, 1::2, :] = w1[:, intermediate_size:, :]
                yield (
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    token_indices,
                    expert_indices,
                    router_scores,
                    w1_sonic,
                )

    def sonicmoe_wrapper(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        token_indices,
        expert_indices,
        router_scores,
        w1_sonic,
    ):
        num_experts = w1_sonic.shape[0]
        ref, _ = moe_general_routing_inputs(
            hidden_states,
            router_scores,
            token_indices,
            expert_indices,
            w1_sonic.permute(1, 2, 0),
            None,
            w2.permute(1, 2, 0),
            None,
            num_experts,
            torch.cuda.current_stream().cuda_stream,
            ActivationType.SWIGLU,
            is_inference_mode_enabled=True,
        )
        return ref

    def gems_wrapper(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        token_indices,
        expert_indices,
        router_scores,
        w1_sonic,
    ):
        return flag_gems.fused_experts_impl(
            hidden_states, w1, w2, topk_weights, topk_ids
        )

    bench = FusedMoESonicmoeBenchmark(
        op_name="fused_moe_bf16_gems_vs_sonicmoe",
        torch_op=sonicmoe_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(gems_wrapper)
    bench.run()
