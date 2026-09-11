# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import triton.language as tl

import flag_gems

from . import base

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False


def _dispatch_fused_moe_kernel_config():
    return {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 1,
        "num_warps": 2,
        "num_stages": 3,
    }


def _dispatch_fused_moe_compute_type(dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported dispatch_fused_moe_kernel dtype: {dtype}")


def _torch_dispatch_fused_moe_kernel_wrapper(
    A,
    B,
    C,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
):
    expert_weights = B[topk_ids.to(torch.long)]
    result = torch.einsum("mk,mtnk->mtn", A.float(), expert_weights.float())
    result = result * topk_weights.float().unsqueeze(-1)
    C.copy_(result.to(C.dtype))
    return C


def _gems_dispatch_fused_moe_kernel_wrapper(
    A,
    B,
    C,
    topk_weights,
    topk_ids,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
):
    flag_gems.dispatch_fused_moe_kernel(
        A,
        B,
        C,
        None,
        None,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        topk_weights.size(1),
        _dispatch_fused_moe_kernel_config(),
        compute_type=_dispatch_fused_moe_compute_type(A.dtype),
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )
    return C


class DispatchFusedMoEKernelBenchmark(base.Benchmark):
    """
    Benchmark for the low-level dispatch_fused_moe_kernel routed GEMM.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, output_size, topk)
        self.shapes = [
            (8, 4, 64, 64, 2),
            (32, 8, 128, 128, 2),
        ]
        self.shape_desc = "num_tokens, num_experts, hidden_size, output_size, topk"

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._dispatch_input_fn(config, cur_dtype)

    def _dispatch_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, output_size, topk = config
        device = flag_gems.device
        kernel_config = _dispatch_fused_moe_kernel_config()

        torch.manual_seed(0)

        A = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * (
            1.0 / hidden_size**0.5
        )
        B = torch.randn(
            num_experts, output_size, hidden_size, device=device, dtype=dtype
        )
        C = torch.empty(num_tokens, topk, output_size, device=device, dtype=dtype)

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype).contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()

        (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
        ) = flag_gems.moe_align_block_size(
            topk_ids,
            kernel_config["BLOCK_SIZE_M"],
            num_experts,
        )

        yield (
            A,
            B,
            C,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
        )


class FusedMoEBenchmark(base.Benchmark):
    """
    Benchmark for fused_experts_impl comparing FlagGems Triton kernel vs vLLM.

    Measures latency of the full fused MoE pipeline:
      moe_align_block_size → GEMM1(up+gate) → SiLU+Mul → GEMM2(down) → moe_sum
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
            yield from self._fused_moe_input_fn(config, cur_dtype)

    def _fused_moe_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        w1 = torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        w2 = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (hidden_states, w1, w2, topk_weights, topk_ids)


def _vllm_fused_moe_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
    """Wrapper to call vllm fused_experts_impl."""
    return vllm_fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation="silu",
    )


def _gems_fused_moe_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
    """Wrapper to call FlagGems fused_experts_impl."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vLLM not installed")
def test_fused_moe_impl_gems_vs_vllm():
    """
    Benchmark FlagGems fused_experts_impl vs vLLM fused_experts_impl (bf16/fp16).
    """
    bench = FusedMoEBenchmark(
        op_name="fused_experts_impl",
        torch_op=_vllm_fused_moe_wrapper,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.set_gems(_gems_fused_moe_wrapper)
    bench.run()


@pytest.mark.dispatch_fused_moe_kernel
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_dispatch_fused_moe_kernel():
    """
    Benchmark FlagGems dispatch_fused_moe_kernel vs a PyTorch routed GEMM reference.
    """
    bench = DispatchFusedMoEKernelBenchmark(
        op_name="dispatch_fused_moe_kernel",
        torch_op=_torch_dispatch_fused_moe_kernel_wrapper,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.set_gems(_gems_dispatch_fused_moe_kernel_wrapper)
    bench.run()
