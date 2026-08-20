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

import importlib
import inspect
import random
from math import ceil

import pytest
import torch
import triton.language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn

from .conftest import QUICK_MODE

random.seed(42)

FUSED_MOE_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (8, 4, 64, 128, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
    # Qwen3.5 shapes (TP=4)
    (1, 256, 2048, 128, 8),
    (10, 256, 2048, 128, 8),
    (256, 256, 2048, 128, 8),
]

if not QUICK_MODE:
    FUSED_MOE_CONFIGS += [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        (4, 16, 512, 1024, 2),
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


FUSED_MOE_QUANT_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]

if not QUICK_MODE:
    FUSED_MOE_QUANT_CONFIGS += [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        # Mixtral-like shapes
        (1, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
    ]

FUSED_MOE_FP8_BLOCKWISE_CONFIGS = list(FUSED_MOE_QUANT_CONFIGS)

if not QUICK_MODE:
    FUSED_MOE_FP8_BLOCKWISE_CONFIGS += [
        # Qwen3.5-397B-A17B
        (1, 512, 4096, 1024, 10),
        (4, 512, 4096, 1024, 10),
        (16, 512, 4096, 1024, 10),
        (64, 512, 4096, 1024, 10),
        (128, 512, 4096, 1024, 10),
        (256, 512, 4096, 1024, 10),
    ]


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()


def test_h20_mixtral_m512_exact_configs(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    monkeypatch.setattr(fused_moe, "_is_h20", lambda: True)

    expected = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    w1_shape = (8, 28672, 4096)
    w2_shape = (8, 4096, 14336)
    gemm1 = fused_moe.try_get_optimal_moe_config(
        w1_shape, w2_shape, 2, "bf16", 512, 8, gemm_stage="gemm1"
    )
    gemm2 = fused_moe.try_get_optimal_moe_config(
        w1_shape, w2_shape, 2, "bf16", 512, 8, gemm_stage="gemm2"
    )

    assert gemm1 == expected
    assert gemm2 == expected
    assert gemm1["BLOCK_SIZE_M"] == gemm2["BLOCK_SIZE_M"]

    # Exact plans have stronger precedence than the nearest-M embedded table.
    # This keeps a future YAML addition from silently disabling the measured
    # H20 plan.
    conflicting_embedded = {512: {**expected, "BLOCK_SIZE_M": 128}}
    monkeypatch.setattr(
        fused_moe,
        "get_moe_configs",
        lambda *args, **kwargs: conflicting_embedded,
    )
    selected, is_embedded = fused_moe.try_get_optimal_moe_config(
        w1_shape,
        w2_shape,
        2,
        "bf16",
        512,
        8,
        gemm_stage="gemm1",
        return_is_embedded=True,
    )
    assert selected == expected
    assert is_embedded is False

    # The override is deliberately exact: nearby batches retain the generic
    # heuristic, while this measured plan is shared by BF16 and FP16.
    adjacent = fused_moe._get_h20_exact_config(
        w1_shape, w2_shape, 511, 8, 2, "bf16", "gemm1"
    )
    assert adjacent is None

    fp16 = fused_moe._get_h20_exact_config(
        w1_shape, w2_shape, 512, 8, 2, "fp16", "gemm1"
    )
    assert fp16 == expected
    assert (
        fused_moe._get_h20_exact_config(
            w1_shape,
            (8, 4096, 28672),
            512,
            8,
            2,
            "bf16",
            "gemm1",
        )
        is None
    )


def test_h20_qwen_m1_bf16_exact_configs(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    monkeypatch.setattr(fused_moe, "_is_h20", lambda: True)

    w1_shape = (256, 256, 2048)
    w2_shape = (256, 2048, 128)
    gemm1 = fused_moe.try_get_optimal_moe_config(
        w1_shape, w2_shape, 8, "bf16", 1, 256, gemm_stage="gemm1"
    )
    gemm2 = fused_moe.try_get_optimal_moe_config(
        w1_shape, w2_shape, 8, "bf16", 1, 256, gemm_stage="gemm2"
    )

    assert gemm1 == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
        "PAIR_GATE_UP_DOT": True,
    }
    assert gemm2 == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 2,
        "num_stages": 2,
    }
    assert gemm1["BLOCK_SIZE_M"] == gemm2["BLOCK_SIZE_M"]
    assert (
        fused_moe._get_h20_exact_config(w1_shape, w2_shape, 1, 256, 8, "fp16", "gemm1")
        is None
    )


def test_h20_qwen_flash_next_m64_bf16_exact_configs(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    monkeypatch.setattr(fused_moe, "_is_h20", lambda: True)

    w1_shape = (512, 128, 2048)
    w2_shape = (512, 2048, 64)
    gemm1 = fused_moe.try_get_optimal_moe_config(
        w1_shape, w2_shape, 10, "bf16", 64, 512, gemm_stage="gemm1"
    )
    gemm2 = fused_moe.try_get_optimal_moe_config(
        w1_shape, w2_shape, 10, "bf16", 64, 512, gemm_stage="gemm2"
    )

    assert gemm1 == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    assert gemm2 == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    assert gemm1["BLOCK_SIZE_M"] == gemm2["BLOCK_SIZE_M"]

    # The measured plan must not leak to adjacent capture sizes, dimensions,
    # dtypes, or expert routing policies.
    for M, hidden_size, intermediate_size, topk, dtype in (
        (63, 2048, 64, 10, "bf16"),
        (64, 4096, 64, 10, "bf16"),
        (64, 2048, 128, 10, "bf16"),
        (64, 2048, 64, 8, "bf16"),
        (64, 2048, 64, 10, "fp16"),
    ):
        adjacent_w1 = (512, 2 * intermediate_size, hidden_size)
        adjacent_w2 = (512, hidden_size, intermediate_size)
        assert (
            fused_moe._get_h20_exact_config(
                adjacent_w1, adjacent_w2, M, 512, topk, dtype, "gemm1"
            )
            is None
        )


def test_moe_block_size_m_validation():
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    config = {"BLOCK_SIZE_M": 64}
    fused_moe._validate_moe_block_size_m(config, config, config)

    with pytest.raises(ValueError, match="must use the same BLOCK_SIZE_M"):
        fused_moe._validate_moe_block_size_m(config, config, {"BLOCK_SIZE_M": 128})


def test_plain_half_embedded_config_falls_back_to_legacy_dtype(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    device_name = "NVIDIA_H100_80GB_HBM3"
    legacy = {64: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32}}
    bf16 = {64: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}}
    device_table = {"512,64,None,0,0": legacy}

    monkeypatch.setattr(fused_moe, "_get_device_name", lambda: device_name)
    monkeypatch.setattr(
        fused_moe,
        "get_embedded_moe_configs",
        lambda: ({device_name: device_table}, {}),
    )

    assert fused_moe.get_moe_configs(512, 64, "bf16") is legacy
    assert fused_moe.get_moe_configs(512, 64, "fp16") is legacy
    assert fused_moe.get_moe_configs(512, 64, "fp8_w8a8") is None

    # A dtype-specific entry always has precedence over the shared legacy
    # table, so existing H20 BF16/FP16 tuning remains authoritative.
    device_table["512,64,bf16,0,0"] = bf16
    assert fused_moe.get_moe_configs(512, 64, "bf16") is bf16
    assert fused_moe.get_moe_configs(512, 64, "fp16") is legacy

    # Keep the compatibility fallback scoped to the Qwen/H100 table. Other
    # devices retain their existing dtype-specific/default selection policy.
    monkeypatch.setattr(fused_moe, "_get_device_name", lambda: "NVIDIA_H20")
    monkeypatch.setattr(
        fused_moe,
        "get_embedded_moe_configs",
        lambda: ({"NVIDIA_H20": {"512,64,None,0,0": legacy}}, {}),
    )
    assert fused_moe.get_moe_configs(512, 64, "bf16") is None


DISPATCH_FUSED_MOE_KERNEL_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, output_size, topk)
    (5, 4, 32, 64, 2),
    (17, 6, 48, 96, 3),
]


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


def _dispatch_fused_moe_reference(A, B, topk_weights, topk_ids):
    expert_weights = B[topk_ids.to(torch.long)]
    result = torch.einsum("mk,mtnk->mtn", A.float(), expert_weights.float())
    result = result * topk_weights.float().unsqueeze(-1)
    return result.to(A.dtype)


@pytest.mark.dispatch_fused_moe_kernel
@pytest.mark.parametrize("config", DISPATCH_FUSED_MOE_KERNEL_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_dispatch_fused_moe_kernel_matches_ref(config, dtype):
    """Test the low-level routed GEMM dispatch against a PyTorch reference."""
    num_tokens, num_experts, hidden_size, output_size, topk = config
    device = flag_gems.device
    kernel_config = _dispatch_fused_moe_kernel_config()

    torch.manual_seed(0)

    A = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * (
        1.0 / hidden_size**0.5
    )
    B = torch.randn(num_experts, output_size, hidden_size, device=device, dtype=dtype)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
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

    result = torch.empty(num_tokens, topk, output_size, device=device, dtype=dtype)
    flag_gems.dispatch_fused_moe_kernel(
        A,
        B,
        result,
        None,
        None,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        topk,
        kernel_config,
        compute_type=_dispatch_fused_moe_compute_type(dtype),
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )

    ref = _dispatch_fused_moe_reference(A, B, topk_weights, topk_ids)

    torch_device_fn.synchronize()

    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


def torch_fused_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of fused MoE (no vLLM dependency).

    Computes:
        Y_m = sum_j  A_mj * W2[e_mj] @ SiLU(W1[e_mj] @ H_m)_{:D} ) * (W1[e_mj] @ H_m)_{D:})

    Args:
        hidden_states: (M, K)
        w1: (E, 2D, K)  -- gate + up projection concatenated
        w2: (E, K, D)   -- down projection
        topk_weights: (M, topk)
        topk_ids: (M, topk)
        apply_router_weight_on_input: apply each route's weight before GEMM1

    Returns:
        output: (M, K)
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            hidden = hidden_states[m].to(torch.float32)
            if apply_router_weight_on_input:
                hidden = hidden * weight.to(torch.float32)
            # GEMM1: up-projection  (1, K) @ (K, 2D) -> (1, 2D)
            z = hidden @ w1[e].T.to(torch.float32)
            # SiLU-and-Mul: split into gate and up, apply SwiGLU
            D = z.shape[-1] // 2
            gate = z[:D]
            up = z[D:]
            s = (gate * torch.sigmoid(gate)) * up  # SiLU(gate) * up
            # GEMM2: down-projection  (1, D) @ (D, K) -> (1, K)
            r = s @ w2[e].T.to(torch.float32)
            # Weighted accumulation
            if apply_router_weight_on_input:
                output[m] += r.to(output.dtype)
            else:
                output[m] += (weight.to(torch.float32) * r).to(output.dtype)

    return output


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fused_moe_vs_ref(config, dtype):
    """Test FlagGems fused_moe against a pure PyTorch reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    # Generate inputs with controlled magnitude to avoid numerical blow-up
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    # Pure PyTorch reference (no vLLM dependency)
    ref = torch_fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

    torch_device_fn.synchronize()

    # Fused bf16/fp16 kernels accumulate rounding errors across two GEMMs
    # and an activation; use tolerances proportional to output magnitude.
    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)

    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False

VLLM_FUSED_MOE_KWARGS = {}
if HAS_VLLM_FUSED_MOE:
    vllm_parameters = inspect.signature(vllm_fused_experts_impl).parameters
    if "inplace" in vllm_parameters:
        VLLM_FUSED_MOE_KWARGS["inplace"] = False
    if "activation" in vllm_parameters:
        VLLM_FUSED_MOE_KWARGS["activation"] = "silu"


def _call_vllm_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids):
    return vllm_fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids, **VLLM_FUSED_MOE_KWARGS
    )


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vLLM is required")
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fused_moe_vs_vllm(config, dtype):
    """Test FlagGems fused_moe against a pure PyTorch reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

    # Generate inputs with controlled magnitude to avoid numerical blow-up
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems result
    result = flag_gems.fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids)

    # Reference result
    ref = _call_vllm_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)

    torch_device_fn.synchronize()

    # Fused bf16/fp16 kernels accumulate rounding errors across two GEMMs
    # and an activation; use tolerances proportional to output magnitude.
    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)

    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="FP8 quantization requires NVIDIA Hopper architecture",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_accuracy_fused_moe_fp8(config):
    """Test FlagGems fused_moe with FP8 W8A8 quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create FP8 weights: quantize and store scale
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    # Per-tensor quantization of weights
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10

    # Quantize w1 per-expert
    w1_scales = []
    w1_fp8_list = []
    for e in range(num_experts):
        amax = w1_fp32[e].abs().amax().clamp(min=eps)
        scale = amax / fp8_max
        w1_q = (w1_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        w1_fp8_list.append(w1_q)
        w1_scales.append(scale)
    w1_fp8 = torch.stack(w1_fp8_list)
    w1_scale = torch.tensor(w1_scales, device=device, dtype=torch.float32)

    # Quantize w2 per-expert
    w2_scales = []
    w2_fp8_list = []
    for e in range(num_experts):
        amax = w2_fp32[e].abs().amax().clamp(min=eps)
        scale = amax / fp8_max
        w2_q = (w2_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        w2_fp8_list.append(w2_q)
        w2_scales.append(scale)
    w2_fp8 = torch.stack(w2_fp8_list)
    w2_scale = torch.tensor(w2_scales, device=device, dtype=torch.float32)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems FP8 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference: use the dequantized weights (fp8 → float) for reference
    w1_deq = torch.zeros_like(w1_fp32).to(dtype)
    for e in range(num_experts):
        w1_deq[e] = (w1_fp8[e].float() * w1_scales[e]).to(dtype)
    w2_deq = torch.zeros_like(w2_fp32).to(dtype)
    for e in range(num_experts):
        w2_deq[e] = (w2_fp8[e].float() * w2_scales[e]).to(dtype)

    ref = torch_fused_moe_quantized_reference(
        hidden_states, w1_deq, w2_deq, topk_weights, topk_ids, quant_mode="fp8"
    )

    torch_device_fn.synchronize()

    # FP8 quantization introduces more error than bf16, use wider tolerances.
    # Two quantized GEMMs + activation create cumulative rounding error.
    rtol = 5e-1
    atol = max(2e-1, ref.abs().max().item() * 1e-1)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


def _fake_quantize_fp8(tensor: torch.Tensor):
    """Simulate FP8 E4M3 quantization round-trip for reference computation."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10
    # Per-tensor quantization
    amax = tensor.abs().amax().clamp(min=eps).float()
    scale = amax / fp8_max
    q = (tensor.float() / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    return q.float() * scale  # dequantized


def _fake_quantize_int8(tensor: torch.Tensor):
    """Simulate INT8 quantization round-trip for reference computation."""
    eps = 1e-10
    # Per-token quantization
    amax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=eps).float()
    scale = amax / 127.0
    q = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)

    return q.float() * scale  # dequantized


def torch_fused_moe_quantized_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_mode: str = "fp8",
) -> torch.Tensor:
    """Reference fused MoE with simulated quantization noise.

    Simulates the quantization → dequantization round-trip on activations
    to model the same numerical behavior as the quantized kernel path.
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    if quant_mode == "fp8":
        fake_quant = _fake_quantize_fp8
    else:
        fake_quant = _fake_quantize_int8

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # Quantize activation before GEMM1
            h_q = fake_quant(hidden_states[m].unsqueeze(0)).squeeze(0)
            # GEMM1
            z = h_q.float() @ w1[e].T.float()
            # SiLU-and-Mul
            D = z.shape[-1] // 2
            gate, up = z[:D], z[D:]
            s = (gate * torch.sigmoid(gate)) * up
            # Quantize intermediate before GEMM2
            s_q = fake_quant(s.unsqueeze(0)).squeeze(0)
            # GEMM2
            r = s_q.float() @ w2[e].T.float()
            output[m] += (weight.float() * r).to(output.dtype)

    return output


def torch_w8a8_block_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
    compute_type: torch.dtype = torch.float32,
) -> torch.Tensor:
    a = a.to(compute_type)
    b = b.to(compute_type)
    assert a.shape[-1] == b.shape[-1]
    assert b.ndim == 2 and b.is_contiguous() and b_scales.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size
    assert (a.shape[-1] + block_k - 1) // block_k == a_scales.shape[-1]
    assert a.shape[:-1] == a_scales.shape[:-1]

    m = a.numel() // a.shape[-1]
    n, k = b.shape
    origin_c_shape = a.shape[:-1] + (n,)
    a = a.reshape(m, a.shape[-1])
    a_scales = a_scales.reshape(m, a_scales.shape[-1])
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == b_scales.shape[0]
    assert k_tiles == b_scales.shape[1]

    c = torch.zeros((m, n), dtype=compute_type, device=a.device)
    a_tiles = [a[:, i * block_k : min((i + 1) * block_k, k)] for i in range(k_tiles)]
    b_tiles = [
        [
            b[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    c_tiles = [c[:, j * block_n : min((j + 1) * block_n, n)] for j in range(n_tiles)]
    a_scale_tiles = [a_scales[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            scale = a_scale_tiles[i] * b_scales[j][i]
            c_tiles[j][:, :] += torch.matmul(a_tiles[i], b_tiles[j][i].t()) * scale

    return c.reshape(origin_c_shape).to(output_dtype)


def torch_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
):
    assert x.shape[-1] % group_size == 0
    assert x.is_contiguous()

    finfo = torch.finfo(dtype)
    x_reshaped = x.reshape(x.numel() // group_size, group_size)
    amax = (
        x_reshaped.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    )
    x_scales = amax / finfo.max
    x_quant = (x_reshaped / x_scales).clamp(min=finfo.min, max=finfo.max).to(dtype)
    x_quant = x_quant.reshape(x.shape)
    x_scales = x_scales.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_quant, x_scales


def torch_w8a8_block_fp8_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    block_shape: list[int],
):
    batch_size, hidden_size = hidden_states.shape
    topk = topk_ids.size(1)
    expanded_hidden = hidden_states.view(batch_size, -1, hidden_size).repeat(1, topk, 1)
    expanded_hidden = expanded_hidden.reshape(-1, hidden_size)
    out = torch.zeros(
        batch_size * topk,
        w2.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    flat_weights = topk_weights.view(-1)
    flat_ids = topk_ids.view(-1)
    _, block_k = block_shape
    hidden_q, hidden_scale = torch_per_token_group_quant_fp8(expanded_hidden, block_k)
    hidden_q = hidden_q.to(torch.float32)

    def silu_and_mul(x):
        import torch.nn.functional as F

        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    for expert_idx in range(w1.shape[0]):
        mask = flat_ids == expert_idx
        if mask.sum():
            inter = torch_w8a8_block_matmul(
                hidden_q[mask],
                w1[expert_idx],
                hidden_scale[mask],
                w1_scale[expert_idx],
                block_shape,
                output_dtype=hidden_states.dtype,
            )
            act = silu_and_mul(inter)
            act_q, act_scale = torch_per_token_group_quant_fp8(act, block_k)
            out[mask] = torch_w8a8_block_matmul(
                act_q,
                w2[expert_idx],
                act_scale,
                w2_scale[expert_idx],
                block_shape,
                output_dtype=hidden_states.dtype,
            )

    return (
        out.view(batch_size, -1, w2.shape[1])
        * flat_weights.view(batch_size, -1, 1).to(out.dtype)
    ).sum(dim=1)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_FP8_BLOCKWISE_CONFIGS)
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="FP8 blockwise quantization requires NVIDIA Hopper architecture",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fused_moe_fp8_blockwise(config, block_shape):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    if hidden_size % block_shape[1] != 0:
        # Invalid shape for block-wise quantization
        return
    if intermediate_size % block_shape[0] != 0:
        # Invalid shape for block-wise quantization
        return

    device = flag_gems.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1_fp8 = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.float32,
        )
        * (1.0 / hidden_size**0.5)
    ).to(torch.float8_e4m3fn)
    w2_fp8 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.float32,
        )
        * (1.0 / intermediate_size**0.5)
    ).to(torch.float8_e4m3fn)

    w1_scale = torch.randn(
        num_experts,
        ceil(intermediate_size * 2 / block_shape[0]),
        ceil(hidden_size / block_shape[1]),
        device=device,
        dtype=torch.float32,
    )
    w2_scale = torch.randn(
        num_experts,
        ceil(hidden_size / block_shape[0]),
        ceil(intermediate_size / block_shape[1]),
        device=device,
        dtype=torch.float32,
    )

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )

    ref = torch_w8a8_block_fp8_moe(
        hidden_states,
        w1_fp8,
        w2_fp8,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        block_shape,
    )

    torch_device_fn.synchronize()

    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fused_moe_int8(config):
    """Test FlagGems fused_moe with INT8 W8A8 per-channel quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT8 weights: quantize per-channel (per output column of each expert)
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10

    # Per-channel quantization of weights: scale per [expert, output_dim]
    # w1 shape: [E, 2D, K] → scale shape: [E, 2D, 1]
    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / 127.0
    w1_int8 = (w1_fp32 / w1_scale_full).round().clamp(-128, 127).to(torch.int8)
    # For the kernel: w1_scale shape [E, 2D] (per-channel: one scale per output dim)
    w1_scale = w1_scale_full.squeeze(-1)

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / 127.0
    w2_int8 = (w2_fp32 / w2_scale_full).round().clamp(-128, 127).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems INT8 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_int8,
        w2_int8,
        topk_weights,
        topk_ids,
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference: use dequantized weights
    w1_deq = (w1_int8.float() * w1_scale_full).to(dtype)
    w2_deq = (w2_int8.float() * w2_scale_full).to(dtype)

    ref = torch_fused_moe_quantized_reference(
        hidden_states, w1_deq, w2_deq, topk_weights, topk_ids, quant_mode="int8"
    )

    torch_device_fn.synchronize()

    # INT8 quantization introduces more error, use wider tolerances
    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 2e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


def torch_fused_moe_weight_only_reference(
    hidden_states: torch.Tensor,
    w1_int: torch.Tensor,
    w2_int: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference fused MoE for weight-only quantization.

    Weights are dequantized (w_int * scale) then used in FP computation.
    Activations remain in original precision (no activation quantization).
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # Dequantize weights
            w1_deq = w1_int[e].float() * w1_scale[e].unsqueeze(-1).float()
            w2_deq = w2_int[e].float() * w2_scale[e].unsqueeze(-1).float()
            # GEMM1
            z = hidden_states[m].float() @ w1_deq.T
            # SiLU-and-Mul
            D = z.shape[-1] // 2
            gate, up = z[:D], z[D:]
            s = (gate * torch.sigmoid(gate)) * up
            # GEMM2
            r = s @ w2_deq.T
            output[m] += (weight.float() * r).to(output.dtype)

    return output


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fused_moe_int8_w8a16(config):
    """Test FlagGems fused_moe with INT8 W8A16 (weight-only) quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT8 weights per-channel
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10
    # Per-channel quantization
    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / 127.0
    w1_int8 = (w1_fp32 / w1_scale_full).round().clamp(-128, 127).to(torch.int8)
    w1_scale = w1_scale_full.squeeze(-1)  # [E, 2D]

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / 127.0
    w2_int8 = (w2_fp32 / w2_scale_full).round().clamp(-128, 127).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)  # [E, K]

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems INT8 W8A16 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_int8,
        w2_int8,
        topk_weights,
        topk_ids,
        use_int8_w8a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference
    ref = torch_fused_moe_weight_only_reference(
        hidden_states,
        w1_int8,
        w2_int8,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )

    torch_device_fn.synchronize()

    # Weight-only quantization has less error than W8A8 since activations
    # are full precision, but still has weight quantization rounding error.
    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 2e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fused_moe_int4_w4a16(config):
    """Test FlagGems fused_moe with INT4 W4A16 (weight-only) quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT4 weights stored in INT8 containers, per-channel
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.float32
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10
    int4_max = 7
    int4_min = -8

    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / int4_max
    w1_int4 = (w1_fp32 / w1_scale_full).round().clamp(int4_min, int4_max).to(torch.int8)
    w1_scale = w1_scale_full.squeeze(-1)

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / int4_max
    w2_int4 = (w2_fp32 / w2_scale_full).round().clamp(int4_min, int4_max).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # FlagGems INT4 W4A16 result
    result = flag_gems.fused_experts_impl(
        hidden_states,
        w1_int4,
        w2_int4,
        topk_weights,
        topk_ids,
        use_int4_w4a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference
    ref = torch_fused_moe_weight_only_reference(
        hidden_states,
        w1_int4,
        w2_int4,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )

    torch_device_fn.synchronize()

    # INT4 has coarser quantization → wider tolerance
    rtol = 3e-1
    atol = max(1e-1, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


def _make_fused_moe_workspace_inputs(dtype):
    device = flag_gems.device
    num_tokens, num_experts, hidden_size, intermediate_size, topk = (
        4,
        8,
        64,
        128,
        2,
    )
    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * (hidden_size**-0.5)
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=dtype,
    ) * (intermediate_size**-0.5)
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype)
    return hidden_states, w1, w2, topk_weights, topk_ids


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="caller-owned fused MoE workspace is currently a generic NVIDIA path",
)
def test_fused_moe_caller_owned_output_and_workspaces(dtype):
    args = _make_fused_moe_workspace_inputs(dtype)
    hidden_states, w1, w2, _topk_weights, topk_ids = args
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.size(1)
    gate_up_size = w1.size(1)
    intermediate_size = gate_up_size // 2

    reference = flag_gems.fused_experts_impl(*args)
    cache13 = torch.empty(
        num_tokens * topk * max(gate_up_size, hidden_size),
        device=hidden_states.device,
        dtype=dtype,
    )
    cache2 = torch.empty(
        num_tokens * topk * intermediate_size,
        device=hidden_states.device,
        dtype=dtype,
    )
    output = torch.empty_like(hidden_states)

    result = flag_gems.fused_experts_impl(
        *args,
        output=output,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2,
    )
    torch_device_fn.synchronize()
    assert result is output
    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)

    # Reusing exactly the same buffers must not retain state from the prior
    # invocation (important for CUDA Graph replay and workspace managers).
    output.fill_(float("nan"))
    repeated = flag_gems.fused_experts_impl(
        *args,
        output=output,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2,
    )
    torch_device_fn.synchronize()
    assert repeated is output
    torch.testing.assert_close(repeated, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="caller-owned fused MoE workspace is currently a generic NVIDIA path",
)
def test_fused_moe_output_may_alias_cache2_for_one_chunk(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    args = _make_fused_moe_workspace_inputs(torch.bfloat16)
    hidden_states, w1, w2, _topk_weights, topk_ids = args
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.size(1)
    gate_up_size = w1.size(1)
    intermediate_size = gate_up_size // 2

    reference = flag_gems.fused_experts_impl(*args)
    cache13 = torch.empty(
        num_tokens * topk * max(gate_up_size, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    common = torch.empty(
        max(
            hidden_states.numel(),
            num_tokens * topk * intermediate_size,
        ),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output = common[: hidden_states.numel()].view_as(hidden_states)

    # Force the direct-sum size threshold low enough that this shape would
    # otherwise use it. cache2/output aliasing must keep the regular GEMM2 +
    # moe_sum ordering because direct-sum would read and write one buffer.
    monkeypatch.setattr(fused_moe, "MOE_DIRECT_SUM_MIN_TOKENS", 1)
    monkeypatch.setattr(fused_moe, "get_moe_configs", lambda *args, **kwargs: None)
    direct_sum_flags = []
    original_dispatch = fused_moe.dispatch_fused_moe_kernel

    def dispatch_spy(*dispatch_args, **dispatch_kwargs):
        direct_sum_flags.append(dispatch_kwargs.get("direct_sum", False))
        return original_dispatch(*dispatch_args, **dispatch_kwargs)

    monkeypatch.setattr(fused_moe, "dispatch_fused_moe_kernel", dispatch_spy)
    result = fused_moe.fused_experts_impl(
        *args,
        output=output,
        intermediate_cache13=cache13,
        intermediate_cache2=common,
    )
    torch_device_fn.synchronize()

    assert result is output
    assert not any(direct_sum_flags)
    torch.testing.assert_close(result, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="caller-owned fused MoE workspace is currently a generic NVIDIA path",
)
def test_fused_moe_rejects_unsafe_workspace_aliases():
    args = _make_fused_moe_workspace_inputs(torch.bfloat16)
    hidden_states, w1, w2, _topk_weights, topk_ids = args
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.size(1)
    gate_up_size = w1.size(1)
    intermediate_size = gate_up_size // 2
    cache13_numel = num_tokens * topk * max(gate_up_size, hidden_size)
    cache2_numel = num_tokens * topk * intermediate_size

    output = torch.empty_like(hidden_states)
    cache13 = torch.empty(
        cache13_numel, device=hidden_states.device, dtype=hidden_states.dtype
    )
    cache2 = torch.empty(
        cache2_numel, device=hidden_states.device, dtype=hidden_states.dtype
    )

    with pytest.raises(ValueError, match="inplace=True and output"):
        flag_gems.fused_experts_impl(*args, inplace=True, output=output)

    with pytest.raises(ValueError, match="intermediate_cache13 is too small"):
        flag_gems.fused_experts_impl(
            *args,
            intermediate_cache13=cache13[:-1],
            intermediate_cache2=cache2,
        )

    shared_scratch = torch.empty(
        max(cache13_numel, cache2_numel),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    with pytest.raises(ValueError, match="must not overlap"):
        flag_gems.fused_experts_impl(
            *args,
            intermediate_cache13=shared_scratch,
            intermediate_cache2=shared_scratch,
        )

    output_cache13 = cache13[: hidden_states.numel()].view_as(hidden_states)
    with pytest.raises(
        ValueError, match="output must not overlap intermediate_cache13"
    ):
        flag_gems.fused_experts_impl(
            *args,
            output=output_cache13,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
        )

    noncontiguous_cache2 = torch.empty(
        cache2_numel * 2,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )[::2]
    with pytest.raises(ValueError, match="intermediate_cache2 must be contiguous"):
        flag_gems.fused_experts_impl(
            *args,
            intermediate_cache13=cache13,
            intermediate_cache2=noncontiguous_cache2,
        )


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="caller-owned fused MoE workspace is currently a generic NVIDIA path",
)
def test_fused_moe_rejects_multichunk_output_cache2_alias():
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    device = flag_gems.device
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_size, intermediate_size, topk = (
        fused_moe.FUSED_MOE_CHUNK_SIZE + 1,
        8,
        16,
        8,
        2,
    )
    hidden_states = torch.zeros((num_tokens, hidden_size), device=device, dtype=dtype)
    w1 = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    )
    w2 = torch.zeros(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=dtype,
    )
    topk_weights = torch.full(
        (num_tokens, topk), 1.0 / topk, device=device, dtype=dtype
    )
    topk_ids = torch.zeros((num_tokens, topk), device=device, dtype=torch.int64)
    max_chunk = fused_moe.FUSED_MOE_CHUNK_SIZE
    cache13 = torch.empty(
        max_chunk * topk * max(2 * intermediate_size, hidden_size),
        device=device,
        dtype=dtype,
    )
    common = torch.empty(
        max(
            hidden_states.numel(),
            max_chunk * topk * intermediate_size,
        ),
        device=device,
        dtype=dtype,
    )
    output = common[: hidden_states.numel()].view_as(hidden_states)

    with pytest.raises(ValueError, match="only for a single MoE chunk"):
        flag_gems.fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            output=output,
            intermediate_cache13=cache13,
            intermediate_cache2=common,
        )


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize(
    "config",
    [
        (1, 256, 2048, 128, 8),
        (4, 8, 128, 256, 2),
        (16, 8, 256, 512, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fused_moe_inplace(config, dtype):
    """Test that inplace=True writes output into hidden_states."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

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

    # Non-inplace reference
    ref = flag_gems.fused_experts_impl(
        hidden_states.clone(),
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
    )

    # Inplace result
    hidden_copy = hidden_states.clone()
    result = flag_gems.fused_experts_impl(
        hidden_copy,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
    )

    torch_device_fn.synchronize()

    # Result should be the same tensor as input
    assert result.data_ptr() == hidden_copy.data_ptr(), "inplace should reuse input"
    torch.testing.assert_close(result, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize(
    "config",
    [
        (1, 256, 2048, 128, 8),
        (4, 8, 128, 256, 2),
        (16, 8, 256, 512, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fused_moe_apply_router_weight_on_input(config, dtype):
    """Test apply_router_weight_on_input vs default (weight on output)."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    torch.manual_seed(0)

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

    # Default (weight on GEMM2 output)
    result_default = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
    )

    # Weight on GEMM1 input
    result_on_input = flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=True,
    )

    torch_device_fn.synchronize()

    # Due to SiLU nonlinearity, these will differ, but both should be
    # close to the reference with weight on the respective path.
    ref = torch_fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)
    ref_on_input = torch_fused_moe_reference(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=True,
    )

    # The default (weight on output) should match our standard reference
    rtol = 1e-1
    atol = max(1e-2, ref.abs().max().item() * 1e-5)
    torch.testing.assert_close(result_default, ref, rtol=rtol, atol=atol)

    atol_on_input = max(1e-2, ref_on_input.abs().max().item() * 1e-5)
    torch.testing.assert_close(
        result_on_input, ref_on_input, rtol=rtol, atol=atol_on_input
    )
