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
        "PERSISTENT_GRID_SIZE": 546,
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


def test_hopper_ep_decode_configs_are_narrow(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    monkeypatch.setattr(fused_moe, "_is_nvidia_sm90", lambda *_: True)
    expert_map = torch.full((288,), -1, dtype=torch.int32)
    expert_map[:18] = torch.arange(18, dtype=torch.int32)

    gemm1 = fused_moe._get_ep_decode_config(
        96, 18, 288, 8, 4096, 2048, 10.0, "bf16", expert_map, "gemm1"
    )
    gemm2 = fused_moe._get_ep_decode_config(
        96, 18, 288, 8, 4096, 2048, 10.0, "bf16", expert_map, "gemm2"
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
        "num_stages": 4,
    }
    assert gemm1["BLOCK_SIZE_M"] == gemm2["BLOCK_SIZE_M"]

    m1_gemm1 = fused_moe._get_ep_decode_config(
        1, 18, 288, 8, 4096, 2048, 10.0, "bf16", expert_map, "gemm1"
    )
    m1_gemm2 = fused_moe._get_ep_decode_config(
        1, 18, 288, 8, 4096, 2048, 10.0, "bf16", expert_map, "gemm2"
    )
    assert m1_gemm1 == gemm1
    assert m1_gemm2 == gemm2
    assert (
        fused_moe._get_ep_decode_config(
            1, 18, 288, 8, 4096, 1280, 10.0, "bf16", expert_map, "gemm1"
        )
        == gemm1
    )
    assert (
        fused_moe._get_ep_decode_config(
            2, 18, 288, 8, 4096, 2048, 10.0, "bf16", expert_map, "gemm2"
        )
        == gemm2
    )

    for M, local_e, global_e, topk, hidden, intermediate, clamp, dtype, mapping in (
        (129, 18, 288, 8, 4096, 2048, 10.0, "bf16", expert_map),
        (96, 16, 288, 8, 4096, 2048, 10.0, "bf16", expert_map),
        (96, 18, 256, 8, 4096, 2048, 10.0, "bf16", expert_map),
        (96, 18, 288, 4, 4096, 2048, 10.0, "bf16", expert_map),
        (96, 18, 288, 8, 6144, 2048, 10.0, "bf16", expert_map),
        (96, 18, 288, 8, 4096, 4096, 10.0, "bf16", expert_map),
        (96, 18, 288, 8, 4096, 2048, None, "bf16", expert_map),
        (96, 18, 288, 8, 4096, 2048, 7.0, "bf16", expert_map),
        (96, 18, 288, 8, 4096, 2048, 10.0, "fp16", expert_map),
        (96, 18, 288, 8, 4096, 2048, 10.0, "bf16", None),
    ):
        assert (
            fused_moe._get_ep_decode_config(
                M,
                local_e,
                global_e,
                topk,
                hidden,
                intermediate,
                clamp,
                dtype,
                mapping,
                "gemm1",
            )
            is None
        )


def test_fused_clamped_swiglu_gate_is_narrow():
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    ep_plan = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}
    compatible = {
        "w1_bias": None,
        "apply_router_weight_on_input": False,
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "ocp_mx_scheme": None,
        "block_shape": None,
    }
    assert fused_moe._should_use_fused_clamped_swiglu(ep_plan, **compatible)

    incompatible_cases = (
        ("missing_ep_plan", None, {}),
        ("w1_bias", ep_plan, {"w1_bias": torch.empty(0)}),
        (
            "router_weight_on_input",
            ep_plan,
            {"apply_router_weight_on_input": True},
        ),
        ("fp8_w8a8", ep_plan, {"use_fp8_w8a8": True}),
        ("int8_w8a8", ep_plan, {"use_int8_w8a8": True}),
        ("int8_w8a16", ep_plan, {"use_int8_w8a16": True}),
        ("int4_w4a16", ep_plan, {"use_int4_w4a16": True}),
        ("ocp_mx", ep_plan, {"ocp_mx_scheme": "mxfp4"}),
        ("block_quant", ep_plan, {"block_shape": [128, 128]}),
    )
    for name, candidate_plan, overrides in incompatible_cases:
        candidate = {**compatible, **overrides}
        assert not fused_moe._should_use_fused_clamped_swiglu(
            candidate_plan, **candidate
        ), name


def test_ep_single_token_route_policy_is_intermediate_size_specific(
    monkeypatch,
):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    ep_plan = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64}
    expert_map = torch.arange(18, dtype=torch.int32)

    assert fused_moe._should_use_ep_naive_route(ep_plan, expert_map, 1, 1280, True)
    assert not fused_moe._should_use_ep_m1_i2048_local_rank(
        ep_plan, expert_map, 1, 1280, True
    )
    assert not fused_moe._should_use_ep_route_block(ep_plan, expert_map, 1, 2048, True)
    assert fused_moe._should_use_ep_naive_route(ep_plan, expert_map, 1, 2048, True)
    assert fused_moe._should_use_ep_m1_i2048_local_rank(
        ep_plan, expert_map, 1, 2048, True
    )
    assert not fused_moe._should_use_ep_route_block(ep_plan, expert_map, 1, 1280, True)
    for intermediate_size in (1279, 1281, 2047, 2049):
        assert not fused_moe._should_use_ep_naive_route(
            ep_plan, expert_map, 1, intermediate_size, True
        )
        assert not fused_moe._should_use_ep_m1_i2048_local_rank(
            ep_plan, expert_map, 1, intermediate_size, True
        )
        assert not fused_moe._should_use_ep_route_block(
            ep_plan, expert_map, 1, intermediate_size, True
        )
    monkeypatch.setattr(fused_moe, "_ENABLE_EXPERIMENTAL_EP_ROUTE_BLOCK", True)
    assert not fused_moe._should_use_ep_m1_i2048_local_rank(
        ep_plan, expert_map, 1, 2048, True
    )
    assert not fused_moe._should_use_ep_naive_route(ep_plan, expert_map, 1, 2048, True)
    assert fused_moe._should_use_ep_route_block(ep_plan, expert_map, 1, 2048, True)

    for helper, intermediate_size in (
        (fused_moe._should_use_ep_naive_route, 1280),
        (fused_moe._should_use_ep_m1_i2048_local_rank, 2048),
        (fused_moe._should_use_ep_route_block, 2048),
    ):
        assert not helper(ep_plan, expert_map, 2, intermediate_size, True)
        assert not helper(ep_plan, expert_map, 1, intermediate_size, False)
        assert not helper(None, expert_map, 1, intermediate_size, True)
        assert not helper(ep_plan, None, 1, intermediate_size, True)


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


@pytest.mark.dispatch_fused_moe_kernel
@pytest.mark.parametrize(
    ("use_expert_map", "skip_invalid_experts"),
    [(True, False), (False, True)],
)
def test_dispatch_wna16_rejects_ep_naive_arguments(
    use_expert_map,
    skip_invalid_experts,
):
    A = torch.empty((1, 16), dtype=torch.bfloat16)
    B = torch.empty((1, 16, 16), dtype=torch.int8)
    C = torch.empty((1, 1, 16), dtype=torch.bfloat16)
    topk_weights = torch.ones((1, 1), dtype=torch.bfloat16)
    sorted_token_ids = torch.zeros(16, dtype=torch.int32)
    expert_ids = torch.zeros(1, dtype=torch.int32)
    num_tokens_post_padded = torch.ones(1, dtype=torch.int32)
    expert_map = torch.zeros(1, dtype=torch.int32) if use_expert_map else None
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 16,
        "GROUP_SIZE_M": 1,
        "num_warps": 1,
        "num_stages": 2,
    }

    with pytest.raises(ValueError, match="WNA16.*does not support"):
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
            1,
            config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=True,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=[0, 16],
            expert_map=expert_map,
            skip_invalid_experts=skip_invalid_experts,
        )


@pytest.mark.dispatch_fused_moe_kernel
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia",
    reason="The persistent H20 configuration is NVIDIA-specific",
)
def test_dispatch_persistent_fused_moe_matches_regular(monkeypatch):
    """Persistent scheduling must preserve the plain-BF16 GEMM2 result."""
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    persistent_launch = fused_moe.invoke_fused_moe_persistent_triton_kernel
    persistent_launches = 0

    def counted_persistent_launch(*args, **kwargs):
        nonlocal persistent_launches
        persistent_launches += 1
        return persistent_launch(*args, **kwargs)

    monkeypatch.setattr(
        fused_moe,
        "invoke_fused_moe_persistent_triton_kernel",
        counted_persistent_launch,
    )
    device = flag_gems.device
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_size, output_size, topk = (16, 32, 70, 130, 4)
    torch.manual_seed(20260824)

    route_inputs = torch.randn(
        num_tokens * topk,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    weights = torch.randn(
        num_experts,
        output_size,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    gating = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype)
    topk_ids = topk_ids.to(torch.int32)

    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 2,
    }
    sorted_ids, expert_ids, total = flag_gems.moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        num_experts,
    )
    expert_ids = expert_ids.clone()
    expert_ids[0] = -1
    regular = torch.full(
        (num_tokens, topk, output_size),
        1.0,
        device=device,
        dtype=dtype,
    )
    persistent = torch.full_like(regular, 1.0)

    dispatch_args = (
        route_inputs,
        weights,
        None,
        None,
        None,
        topk_weights,
        sorted_ids,
        expert_ids,
        total,
        True,
        1,
    )
    flag_gems.dispatch_fused_moe_kernel(
        dispatch_args[0],
        dispatch_args[1],
        regular,
        *dispatch_args[2:],
        config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )
    # Keep the grid much smaller than the logical tile count so every CTA
    # exercises the persistent stride loop. The odd K/N sizes also cover both
    # reduction and output tails.
    persistent_config = {**config, "PERSISTENT_GRID_SIZE": 7}
    flag_gems.dispatch_fused_moe_kernel(
        dispatch_args[0],
        dispatch_args[1],
        persistent,
        *dispatch_args[2:],
        persistent_config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )
    torch_device_fn.synchronize()

    assert persistent_launches == 1
    assert torch.equal(persistent, regular)


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


def torch_fused_moe_ep_reference(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    expert_map,
    clamp_limit=None,
):
    """Reference one EP rank, skipping routes assigned to remote experts."""
    # moe_sum accumulates all routed outputs in FP32 and casts once. Keep that
    # boundary here so the reference cannot hide route errors behind repeated
    # BF16 accumulation rounding.
    output = torch.zeros_like(hidden_states, dtype=torch.float32)
    for token_idx in range(hidden_states.shape[0]):
        hidden = hidden_states[token_idx].float()
        for route_idx in range(topk_ids.shape[1]):
            global_expert = int(topk_ids[token_idx, route_idx].item())
            if global_expert < 0 or global_expert >= expert_map.numel():
                continue
            local_expert = int(expert_map[global_expert].item())
            if local_expert < 0 or local_expert >= w1.shape[0]:
                continue
            # Match GEMM1's BF16 cache boundary before applying SwiGLU.
            gate_up = (hidden @ w1[local_expert].T.float()).to(w1.dtype).float()
            gate, up = gate_up.chunk(2)
            if clamp_limit is not None:
                gate = gate.clamp(max=clamp_limit)
                up = up.clamp(min=-clamp_limit, max=clamp_limit)
            # Activation and GEMM2 each materialize into BF16 workspaces.
            activated = (gate * torch.sigmoid(gate) * up).to(w1.dtype).float()
            routed = activated @ w2[local_expert].T.float()
            weighted_routed = (routed * topk_weights[token_idx, route_idx].float()).to(
                w2.dtype
            )
            output[token_idx] += weighted_routed.float()
    return output.to(hidden_states.dtype)


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="optimized fused-MoE EP path is enabled only on NVIDIA SM90",
)
def test_fused_moe_ep_m1_naive_route_matches_compact_and_graph(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    plans = {
        stage: config.copy()
        for stage, config in fused_moe._HOPPER_EP_DECODE_PLAN.items()
    }
    monkeypatch.setattr(
        fused_moe,
        "_get_ep_decode_config",
        lambda *args: plans[args[-1]].copy(),
    )

    def force_naive_route(*_args, **_kwargs):
        return True

    monkeypatch.setattr(fused_moe, "_should_use_ep_naive_route", force_naive_route)
    monkeypatch.setattr(
        fused_moe,
        "_should_use_ep_route_block",
        lambda *_args, **_kwargs: False,
    )

    align_calls = 0
    original_align = fused_moe.moe_align_block_size

    def align_spy(*args, **kwargs):
        nonlocal align_calls
        align_calls += 1
        return original_align(*args, **kwargs)

    dispatches = []
    original_dispatch = fused_moe.dispatch_fused_moe_kernel

    def dispatch_spy(*args, **kwargs):
        dispatches.append(
            {
                "expert_map": kwargs.get("expert_map"),
                "skip_invalid_experts": kwargs.get("skip_invalid_experts", False),
                "sorted_token_ids": args[7],
                "direct_sum": kwargs.get("direct_sum", False),
                "config": dict(args[12]),
            }
        )
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(fused_moe, "moe_align_block_size", align_spy)
    monkeypatch.setattr(fused_moe, "dispatch_fused_moe_kernel", dispatch_spy)

    device = flag_gems.device
    dtype = torch.bfloat16
    m, global_e, local_e, hidden_size, intermediate, topk = (1, 288, 18, 64, 32, 8)
    torch.manual_seed(20260824)
    hidden = 4 * torch.randn((m, hidden_size), device=device, dtype=dtype)
    w1 = 0.5 * torch.randn(
        (local_e, 2 * intermediate, hidden_size), device=device, dtype=dtype
    )
    w2 = torch.randn(
        (local_e, hidden_size, intermediate), device=device, dtype=dtype
    ) * (intermediate**-0.5)
    shard_begin = 7 * local_e
    expert_map = torch.full((global_e,), -1, device=device, dtype=torch.int32)
    expert_map[shard_begin : shard_begin + local_e] = torch.arange(
        local_e, device=device, dtype=torch.int32
    )
    topk_ids = torch.tensor(
        [[shard_begin + 3, -1, global_e, 2**40, -(2**40), 50, 240, 287]],
        device=device,
        dtype=torch.int64,
    )
    topk_weights = torch.rand((m, topk), device=device, dtype=torch.float32)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype)

    def run(**workspace_kwargs):
        return flag_gems.fused_experts_impl(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=global_e,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            **workspace_kwargs,
        )

    naive_result = run()
    assert align_calls == 0
    assert len(dispatches) == 2
    for dispatch in dispatches:
        assert dispatch["expert_map"] is expert_map
        assert dispatch["skip_invalid_experts"]
        assert dispatch["sorted_token_ids"] is None

    monkeypatch.setattr(
        fused_moe,
        "_should_use_ep_naive_route",
        lambda *_args, **_kwargs: False,
    )
    compact_result = run()
    assert align_calls == 1
    assert torch.equal(naive_result, compact_result)

    reference = torch_fused_moe_ep_reference(
        hidden, w1, w2, topk_weights, topk_ids, expert_map, clamp_limit=10.0
    )
    torch.testing.assert_close(naive_result, reference, rtol=1e-1, atol=1e-2)

    monkeypatch.setattr(fused_moe, "_should_use_ep_naive_route", force_naive_route)
    cache13 = torch.empty(
        m * topk * max(2 * intermediate, hidden_size), device=device, dtype=dtype
    )
    cache2 = torch.empty(m * topk * intermediate, device=device, dtype=dtype)
    # Exercise the modular vLLM layout where final output aliases the beginning
    # of cache2.  GEMM2 consumes the one local activation before moe_sum_ep
    # overwrites that storage; untouched remote rows must never be observed.
    output = cache2[: m * hidden_size].view(m, hidden_size)

    def graph_op():
        return run(
            output=output,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
        )

    eager = graph_op().clone()
    torch_device_fn.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_result = graph_op()
    graph.replay()
    torch_device_fn.synchronize()
    assert graph_result is output
    assert torch.equal(eager, graph_result)
    assert torch.equal(naive_result, graph_result)


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="fused-MoE EP local-rank path is enabled only on NVIDIA SM90",
)
def test_fused_moe_ep_m1_i2048_local_rank_alias_dynamic_graph(monkeypatch):
    """Exercise the strict production kernel, including dynamic route safety."""
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    torch.manual_seed(20260824)
    device = flag_gems.device
    dtype = torch.bfloat16
    m, global_e, local_e, hidden_size, intermediate, topk = (
        1,
        288,
        18,
        4096,
        2048,
        8,
    )
    shard_begin = 7 * local_e
    hidden = torch.randn((m, hidden_size), device=device, dtype=dtype)
    w1 = torch.empty(
        (local_e, 2 * intermediate, hidden_size), device=device, dtype=dtype
    ).normal_(std=hidden_size**-0.5)
    w2 = torch.empty(
        (local_e, hidden_size, intermediate), device=device, dtype=dtype
    ).normal_(std=intermediate**-0.5)
    topk_weights = torch.rand((m, topk), device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    original_local_rank_gate = fused_moe._should_use_ep_m1_i2048_local_rank
    original_launcher = fused_moe.fused_moe_ep_m1_i2048_local_rank
    launcher_calls = 0

    def launcher_spy(*args, **kwargs):
        nonlocal launcher_calls
        launcher_calls += 1
        return original_launcher(*args, **kwargs)

    monkeypatch.setattr(
        fused_moe,
        "fused_moe_ep_m1_i2048_local_rank",
        launcher_spy,
    )

    def make_workspaces():
        cache13 = torch.empty(
            m * topk * max(2 * intermediate, hidden_size),
            device=device,
            dtype=dtype,
        )
        cache2 = torch.empty(
            m * topk * intermediate,
            device=device,
            dtype=dtype,
        )
        # Match modular vLLM: final output aliases the beginning of cache2.
        output = cache2[: m * hidden_size].view(m, hidden_size)
        return cache13, cache2, output

    def run(ids, expert_map, workspaces, weights=topk_weights, **kwargs):
        cache13, cache2, output = workspaces
        return fused_moe.fused_experts_impl(
            hidden,
            w1,
            w2,
            weights,
            ids,
            global_num_experts=global_e,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            output=output,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
            **kwargs,
        )

    remote_values = (0, 20, 50, 80, 100, 180, 240, 287)
    dtype_pairs = (
        (torch.int32, torch.int32),
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int64, torch.int64),
    )
    for ids_dtype, map_dtype in dtype_pairs:
        expert_map = torch.full((global_e,), -1, device=device, dtype=map_dtype)
        expert_map[shard_begin : shard_begin + local_e] = torch.arange(
            local_e, device=device, dtype=map_dtype
        )
        remote = torch.tensor([remote_values], device=device, dtype=ids_dtype)
        early_local = remote.clone()
        early_local[0, 0] = shard_begin
        late_local = remote.clone()
        late_local[0, 7] = shard_begin
        extreme = 2**40 if ids_dtype == torch.int64 else torch.iinfo(torch.int32).max
        route_inputs = (
            remote,
            early_local,
            late_local,
            torch.tensor(
                [
                    [
                        shard_begin,
                        20,
                        shard_begin + 1,
                        80,
                        shard_begin + 2,
                        180,
                        shard_begin + 3,
                        287,
                    ]
                ],
                device=device,
                dtype=ids_dtype,
            ),
            torch.full((m, topk), shard_begin + 3, device=device, dtype=ids_dtype),
            torch.tensor(
                [[-1, global_e, extreme, -extreme, shard_begin, 143, 144, 287]],
                device=device,
                dtype=ids_dtype,
            ),
        )
        for route_ids in route_inputs:
            reference_workspaces = make_workspaces()
            candidate_workspaces = make_workspaces()
            with monkeypatch.context() as direct_context:
                direct_context.setattr(
                    fused_moe,
                    "_should_use_ep_m1_i2048_local_rank",
                    lambda *_args, **_kwargs: False,
                )
                reference = run(route_ids, expert_map, reference_workspaces).clone()
            candidate = run(route_ids, expert_map, candidate_workspaces).clone()
            assert torch.equal(reference, candidate), (ids_dtype, map_dtype)

    # Capture both policies once with int64 routing, then mutate IDs, map and
    # weights in place. One graph must remain correct for every replay.
    ids = torch.tensor(
        [[shard_begin, 20, 50, 80, 100, 180, 240, 287]],
        device=device,
        dtype=torch.int64,
    )
    expert_map = torch.full((global_e,), -1, device=device, dtype=torch.int64)
    expert_map[shard_begin : shard_begin + local_e] = torch.arange(
        local_e, device=device, dtype=torch.int64
    )
    reference_workspaces = make_workspaces()
    candidate_workspaces = make_workspaces()

    def capture(fn):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch_device_fn.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
        return graph, output

    with monkeypatch.context() as direct_context:
        direct_context.setattr(
            fused_moe,
            "_should_use_ep_m1_i2048_local_rank",
            lambda *_args, **_kwargs: False,
        )
        reference_graph, reference_output = capture(
            lambda: run(ids, expert_map, reference_workspaces)
        )
    candidate_graph, candidate_output = capture(
        lambda: run(ids, expert_map, candidate_workspaces)
    )

    dynamic_routes = (
        torch.tensor([remote_values], device=device, dtype=torch.int64),
        torch.tensor(
            [[0, 20, 50, 80, 100, 180, 240, shard_begin + 7]],
            device=device,
            dtype=torch.int64,
        ),
        torch.arange(
            shard_begin, shard_begin + topk, device=device, dtype=torch.int64
        ).view(m, topk),
        torch.full((m, topk), shard_begin + 3, device=device, dtype=torch.int64),
        torch.tensor(
            [[-1, global_e, 2**40, -(2**40), shard_begin, 143, 144, 287]],
            device=device,
            dtype=torch.int64,
        ),
    )
    for route_update in dynamic_routes:
        ids.copy_(route_update)
        reference_graph.replay()
        candidate_graph.replay()
        torch_device_fn.synchronize()
        assert torch.equal(reference_output, candidate_output)

    ids.copy_(dynamic_routes[1])
    expert_map[shard_begin + 7] = local_e  # invalid mapped local expert
    topk_weights.copy_(torch.flip(topk_weights, dims=(1,)))
    reference_graph.replay()
    candidate_graph.replay()
    torch_device_fn.synchronize()
    assert torch.equal(reference_output, candidate_output)

    # Both FP32 and BF16 router weights are supported. Other weight/bias
    # contracts must not enter the specialization.
    bf16_reference_workspaces = make_workspaces()
    bf16_candidate_workspaces = make_workspaces()
    with monkeypatch.context() as direct_context:
        direct_context.setattr(
            fused_moe,
            "_should_use_ep_m1_i2048_local_rank",
            lambda *_args, **_kwargs: False,
        )
        bf16_reference = run(
            ids,
            expert_map,
            bf16_reference_workspaces,
            weights=topk_weights.to(dtype),
        ).clone()
    calls_before_bf16 = launcher_calls
    bf16_candidate = run(
        ids,
        expert_map,
        bf16_candidate_workspaces,
        weights=topk_weights.to(dtype),
    ).clone()
    assert launcher_calls == calls_before_bf16 + 1
    assert torch.equal(bf16_reference, bf16_candidate)

    calls_before_fallback = launcher_calls
    fallback_workspaces = make_workspaces()
    run(
        ids,
        expert_map,
        fallback_workspaces,
        w2_bias=torch.zeros((local_e, hidden_size), device=device, dtype=dtype),
    )
    assert launcher_calls == calls_before_fallback
    run(
        ids,
        expert_map,
        make_workspaces(),
        weights=topk_weights.half(),
    )
    assert launcher_calls == calls_before_fallback
    assert original_local_rank_gate({"BLOCK_SIZE_M": 16}, expert_map, 1, 2048, True)


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="optimized fused-MoE EP path is enabled only on NVIDIA SM90",
)
def test_fused_moe_ep_m1_route_block_matches_compact_alias_graph(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    plans = {
        stage: config.copy()
        for stage, config in fused_moe._HOPPER_EP_DECODE_PLAN.items()
    }
    monkeypatch.setattr(
        fused_moe,
        "_get_ep_decode_config",
        lambda *args: plans[args[-1]].copy(),
    )
    monkeypatch.setattr(
        fused_moe,
        "_should_use_ep_naive_route",
        lambda *_args, **_kwargs: False,
    )

    def force_route_block(*_args, **_kwargs):
        return bool(_args[4])

    monkeypatch.setattr(fused_moe, "_should_use_ep_route_block", force_route_block)

    route_block_calls = 0
    compact_calls = 0
    original_route_block = fused_moe.moe_align_block_size_ep_route_block
    original_align = fused_moe.moe_align_block_size

    def route_block_spy(*args, **kwargs):
        nonlocal route_block_calls
        route_block_calls += 1
        return original_route_block(*args, **kwargs)

    def align_spy(*args, **kwargs):
        nonlocal compact_calls
        compact_calls += 1
        return original_align(*args, **kwargs)

    dispatches = []
    original_dispatch = fused_moe.dispatch_fused_moe_kernel

    def dispatch_spy(*args, **kwargs):
        dispatches.append(
            {
                "expert_map": kwargs.get("expert_map"),
                "skip_invalid_experts": kwargs.get("skip_invalid_experts", False),
                "sorted_token_ids": args[7],
                "direct_sum": kwargs.get("direct_sum", False),
                "config": dict(args[12]),
            }
        )
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(
        fused_moe, "moe_align_block_size_ep_route_block", route_block_spy
    )
    monkeypatch.setattr(fused_moe, "moe_align_block_size", align_spy)
    monkeypatch.setattr(fused_moe, "dispatch_fused_moe_kernel", dispatch_spy)

    device = flag_gems.device
    dtype = torch.bfloat16
    m, global_e, local_e, hidden_size, intermediate, topk = (1, 288, 18, 64, 32, 8)
    torch.manual_seed(20260824)
    hidden = 4 * torch.randn((m, hidden_size), device=device, dtype=dtype)
    w1 = 0.5 * torch.randn(
        (local_e, 2 * intermediate, hidden_size), device=device, dtype=dtype
    )
    w2 = torch.randn(
        (local_e, hidden_size, intermediate), device=device, dtype=dtype
    ) * (intermediate**-0.5)
    shard_begin = 7 * local_e
    expert_map = torch.full((global_e,), -1, device=device, dtype=torch.int32)
    expert_map[shard_begin : shard_begin + local_e] = torch.arange(
        local_e, device=device, dtype=torch.int32
    )
    late_route_ids = torch.tensor(
        [[-1, global_e, 2**40, -(2**40), 50, 240, 287, shard_begin + 3]],
        device=device,
        dtype=torch.int64,
    )
    early_route_ids = late_route_ids.roll(1, dims=1)
    topk_ids = late_route_ids.clone()
    topk_weights = torch.rand((m, topk), device=device, dtype=torch.float32)
    topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).to(dtype)

    def run(**workspace_kwargs):
        return flag_gems.fused_experts_impl(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=global_e,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
            **workspace_kwargs,
        )

    route_block_result = run()
    assert route_block_calls == 1
    assert compact_calls == 0
    assert len(dispatches) == 2
    for dispatch in dispatches:
        assert dispatch["expert_map"] is None
        assert not dispatch["skip_invalid_experts"]
        assert dispatch["sorted_token_ids"] is not None
        assert not dispatch["direct_sum"]
    assert dispatches[0]["config"]["BLOCK_SIZE_N"] == 16
    assert dispatches[0]["config"]["num_warps"] == 2
    assert dispatches[1]["config"]["BLOCK_SIZE_N"] == 64
    assert dispatches[1]["config"]["num_stages"] == 4

    monkeypatch.setattr(
        fused_moe,
        "_should_use_ep_route_block",
        lambda *_args, **_kwargs: False,
    )
    compact_result = run()
    assert compact_calls == 1
    assert torch.equal(route_block_result, compact_result)

    reference = torch_fused_moe_ep_reference(
        hidden, w1, w2, topk_weights, topk_ids, expert_map, clamp_limit=10.0
    )
    torch.testing.assert_close(route_block_result, reference, rtol=1e-1, atol=1e-2)

    monkeypatch.setattr(fused_moe, "_should_use_ep_route_block", force_route_block)
    fallback_dispatch_start = len(dispatches)
    with monkeypatch.context() as fallback_context:
        fallback_context.setattr(
            fused_moe,
            "_should_use_fused_clamped_swiglu",
            lambda *_args, **_kwargs: False,
        )
        separate_activation_result = run()
    assert torch.equal(route_block_result, separate_activation_result)
    fallback_dispatches = dispatches[fallback_dispatch_start:]
    assert len(fallback_dispatches) == 2
    assert fallback_dispatches[0]["config"]["BLOCK_SIZE_N"] == 64
    assert fallback_dispatches[0]["config"]["num_warps"] == 4
    assert fallback_dispatches[1]["config"]["BLOCK_SIZE_N"] == 128

    # GEMM2 bias does not invalidate the fused GEMM1 activation, but this
    # unmeasured variant must retain compact alignment and the shared tiles.
    route_block_calls_before_bias = route_block_calls
    compact_calls_before_bias = compact_calls
    bias_dispatch_start = len(dispatches)
    w2_bias = torch.zeros((local_e, hidden_size), device=device, dtype=dtype)
    bias_result = run(w2_bias=w2_bias)
    assert torch.equal(route_block_result, bias_result)
    assert route_block_calls == route_block_calls_before_bias
    assert compact_calls == compact_calls_before_bias + 1
    bias_dispatches = dispatches[bias_dispatch_start:]
    assert len(bias_dispatches) == 2
    assert bias_dispatches[0]["config"]["BLOCK_SIZE_N"] == 32
    assert bias_dispatches[0]["config"]["num_warps"] == 4
    assert bias_dispatches[1]["config"]["BLOCK_SIZE_N"] == 128

    monkeypatch.setattr(fused_moe, "_should_use_ep_route_block", force_route_block)
    cache13 = torch.empty(
        m * topk * max(2 * intermediate, hidden_size), device=device, dtype=dtype
    )
    cache2 = torch.empty(m * topk * intermediate, device=device, dtype=dtype)
    output = cache2[: m * hidden_size].view(m, hidden_size)

    def graph_op():
        return run(
            output=output,
            intermediate_cache13=cache13,
            intermediate_cache2=cache2,
        )

    eager_late = graph_op().clone()
    torch_device_fn.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_result = graph_op()
    graph.replay()
    torch_device_fn.synchronize()
    assert graph_result is output
    assert torch.equal(eager_late, graph_result)

    route_updates = (
        torch.tensor(
            [[0, 20, 50, 80, 100, 180, 240, 287]],
            device=device,
            dtype=torch.int64,
        ),
        torch.arange(
            shard_begin, shard_begin + topk, device=device, dtype=torch.int64
        ).view(1, topk),
        torch.full((1, topk), shard_begin + 3, device=device, dtype=torch.int64),
        early_route_ids,
        torch.tensor(
            [[0, 20, 50, 80, 100, 180, 240, 287]],
            device=device,
            dtype=torch.int64,
        ),
    )
    for route_update in route_updates:
        topk_ids.copy_(route_update)
        monkeypatch.setattr(
            fused_moe,
            "_should_use_ep_route_block",
            lambda *_args, **_kwargs: False,
        )
        compact_expected = run().clone()
        monkeypatch.setattr(fused_moe, "_should_use_ep_route_block", force_route_block)
        graph.replay()
        torch_device_fn.synchronize()
        assert torch.equal(compact_expected, graph_result)


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="optimized fused-MoE EP path is enabled only on NVIDIA SM90",
)
def test_fused_moe_ep_m1_policy_does_not_match_only_a_tail_chunk(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    plans = {
        stage: config.copy()
        for stage, config in fused_moe._HOPPER_EP_DECODE_PLAN.items()
    }
    monkeypatch.setattr(fused_moe, "FUSED_MOE_CHUNK_SIZE", 2)
    monkeypatch.setattr(
        fused_moe,
        "_get_ep_decode_config",
        lambda *args: plans[args[-1]].copy(),
    )

    observed_call_sizes = {"local_rank": [], "direct": [], "route_block": []}
    original_local_rank_gate = fused_moe._should_use_ep_m1_i2048_local_rank
    original_direct_gate = fused_moe._should_use_ep_naive_route
    original_route_block_gate = fused_moe._should_use_ep_route_block

    def local_rank_gate_spy(*args, **kwargs):
        observed_call_sizes["local_rank"].append(args[2])
        return original_local_rank_gate(*args, **kwargs)

    def direct_gate_spy(*args, **kwargs):
        observed_call_sizes["direct"].append(args[2])
        return original_direct_gate(*args, **kwargs)

    def route_block_gate_spy(*args, **kwargs):
        observed_call_sizes["route_block"].append(args[2])
        return original_route_block_gate(*args, **kwargs)

    align_calls = 0
    original_align = fused_moe.moe_align_block_size

    def align_spy(*args, **kwargs):
        nonlocal align_calls
        align_calls += 1
        return original_align(*args, **kwargs)

    monkeypatch.setattr(
        fused_moe,
        "_should_use_ep_m1_i2048_local_rank",
        local_rank_gate_spy,
    )
    monkeypatch.setattr(fused_moe, "_should_use_ep_naive_route", direct_gate_spy)
    monkeypatch.setattr(fused_moe, "_should_use_ep_route_block", route_block_gate_spy)
    monkeypatch.setattr(fused_moe, "moe_align_block_size", align_spy)

    device = flag_gems.device
    dtype = torch.bfloat16
    m, global_e, local_e, hidden_size, intermediate, topk = (3, 288, 18, 64, 32, 8)
    torch.manual_seed(20260824)
    hidden = torch.randn((m, hidden_size), device=device, dtype=dtype)
    w1 = torch.randn(
        (local_e, 2 * intermediate, hidden_size), device=device, dtype=dtype
    )
    w2 = torch.randn((local_e, hidden_size, intermediate), device=device, dtype=dtype)
    topk_ids = torch.arange(m * topk, device=device, dtype=torch.int32).view(m, topk)
    topk_weights = torch.full((m, topk), 1 / topk, device=device, dtype=dtype)
    expert_map = torch.full((global_e,), -1, device=device, dtype=torch.int32)
    expert_map[:local_e] = torch.arange(local_e, device=device, dtype=torch.int32)

    result = flag_gems.fused_experts_impl(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=global_e,
        expert_map=expert_map,
        gemm1_clamp_limit=10.0,
    )
    assert result.shape == hidden.shape
    assert align_calls == 2
    assert observed_call_sizes == {
        "local_rank": [m, m],
        "direct": [m, m],
        "route_block": [m, m],
    }


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="optimized fused-MoE EP path is enabled only on NVIDIA SM90",
)
def test_fused_moe_ep_matches_reference(monkeypatch):
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    align = importlib.import_module("flag_gems.fused.moe_align_block_size")

    # Exercise the integrated optimized path with a small test tensor: the
    # production selector is deliberately shape-strict and would otherwise
    # reject H=64/I=32 before reaching compact EP alignment.
    plans = {
        stage: config.copy()
        for stage, config in fused_moe._HOPPER_EP_DECODE_PLAN.items()
    }
    monkeypatch.setattr(
        fused_moe,
        "_get_ep_decode_config",
        lambda *args: plans[args[-1]].copy(),
    )
    compact_dispatches = 0
    original_compact = align.moe_align_block_size_ep_compact

    def compact_spy(*args, **kwargs):
        nonlocal compact_dispatches
        compact_dispatches += 1
        return original_compact(*args, **kwargs)

    monkeypatch.setattr(align, "moe_align_block_size_ep_compact", compact_spy)

    device = flag_gems.device
    dtype = torch.bfloat16
    m, global_e, local_e, hidden_size, intermediate, topk = (4, 288, 18, 64, 32, 8)
    torch.manual_seed(20260824)
    hidden = 4 * torch.randn((m, hidden_size), device=device, dtype=dtype)
    w1 = (
        torch.randn(
            (local_e, 2 * intermediate, hidden_size), device=device, dtype=dtype
        )
        * 0.5
    )
    w2 = torch.randn(
        (local_e, hidden_size, intermediate), device=device, dtype=dtype
    ) * (intermediate**-0.5)

    gemm1_dispatches = []
    original_dispatch = fused_moe.dispatch_fused_moe_kernel

    def dispatch_spy(*args, **kwargs):
        if args[1].data_ptr() == w1.data_ptr():
            gemm1_dispatches.append(
                {
                    "config": dict(args[12]),
                    "fuse_silu": kwargs.get("FUSE_SILU", False),
                    "output_shape": tuple(args[2].shape),
                    "weights_is_none": args[6] is None,
                }
            )
        return original_dispatch(*args, **kwargs)

    activation_calls = 0
    original_activation = fused_moe.apply_moe_activation

    def activation_spy(*args, **kwargs):
        nonlocal activation_calls
        activation_calls += 1
        return original_activation(*args, **kwargs)

    monkeypatch.setattr(fused_moe, "dispatch_fused_moe_kernel", dispatch_spy)
    monkeypatch.setattr(fused_moe, "apply_moe_activation", activation_spy)

    shard_begin = 7 * local_e
    expert_map = torch.full((global_e,), -1, device=device, dtype=torch.int32)
    expert_map[shard_begin : shard_begin + local_e] = torch.arange(
        local_e, device=device, dtype=torch.int32
    )
    topk_ids = torch.tensor(
        [
            [shard_begin, 0, shard_begin + 1, 50, 80, 200, 250, 287],
            [1, 2, shard_begin + 3, shard_begin + 4, 90, 190, 270, 280],
            [shard_begin + 17, 3, 30, 60, 100, 180, 240, 286],
            [4, shard_begin + 8, 40, shard_begin + 9, 110, 170, 230, 285],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.rand((m, topk), device=device, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    result = flag_gems.fused_experts_impl(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=global_e,
        expert_map=expert_map,
        gemm1_clamp_limit=10.0,
    )
    cache13 = torch.empty(
        m * topk * max(2 * intermediate, hidden_size), device=device, dtype=dtype
    )
    cache2 = torch.empty(m * topk * intermediate, device=device, dtype=dtype)
    aliased_output = cache2[: m * hidden_size].view(m, hidden_size)
    aliased_result = flag_gems.fused_experts_impl(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=global_e,
        expert_map=expert_map,
        gemm1_clamp_limit=10.0,
        output=aliased_output,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2,
    )
    assert activation_calls == 0
    assert len(gemm1_dispatches) == 2
    for dispatch in gemm1_dispatches:
        config = dispatch["config"]
        assert dispatch["fuse_silu"]
        assert dispatch["weights_is_none"]
        assert dispatch["output_shape"] == (m, topk, intermediate)
        assert config["BLOCK_SIZE_N"] == 32
        assert config["PAIR_GATE_UP_DOT"] is True
        assert config["CLAMPED_BF16_BOUNDARY"] is True
        assert config["CLAMP_LIMIT"] == 10.0

    # The public entry currently asserts activation="silu". Override only the
    # parsed enum so this test can still exercise the defensive caller-side
    # activation gate and prove that GELU never receives the fused epilogue.
    with monkeypatch.context() as activation_context:
        activation_context.setattr(
            fused_moe.MoEActivation,
            "from_str",
            classmethod(lambda cls, _value: cls.GELU),
        )
        flag_gems.fused_experts_impl(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=global_e,
            expert_map=expert_map,
            gemm1_clamp_limit=10.0,
        )
    assert activation_calls == 1
    assert len(gemm1_dispatches) == 3
    gelu_dispatch = gemm1_dispatches[-1]
    assert not gelu_dispatch["fuse_silu"]
    assert not gelu_dispatch["weights_is_none"]
    assert gelu_dispatch["output_shape"] == (m, topk, 2 * intermediate)
    assert gelu_dispatch["config"]["BLOCK_SIZE_N"] == 64
    assert "PAIR_GATE_UP_DOT" not in gelu_dispatch["config"]
    assert "CLAMPED_BF16_BOUNDARY" not in gelu_dispatch["config"]

    monkeypatch.setattr(
        fused_moe,
        "_should_use_fused_clamped_swiglu",
        lambda *args, **kwargs: False,
    )
    unfused_result = flag_gems.fused_experts_impl(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=global_e,
        expert_map=expert_map,
        gemm1_clamp_limit=10.0,
    )
    reference = torch_fused_moe_ep_reference(
        hidden, w1, w2, topk_weights, topk_ids, expert_map, clamp_limit=10.0
    )
    unclamped_reference = torch_fused_moe_ep_reference(
        hidden, w1, w2, topk_weights, topk_ids, expert_map
    )
    torch_device_fn.synchronize()
    assert compact_dispatches == 4
    assert activation_calls == 2
    assert len(gemm1_dispatches) == 4
    unfused_dispatch = gemm1_dispatches[-1]
    assert not unfused_dispatch["fuse_silu"]
    assert not unfused_dispatch["weights_is_none"]
    assert unfused_dispatch["output_shape"] == (m, topk, 2 * intermediate)
    assert unfused_dispatch["config"]["BLOCK_SIZE_N"] == 64
    assert "PAIR_GATE_UP_DOT" not in unfused_dispatch["config"]
    assert "CLAMPED_BF16_BOUNDARY" not in unfused_dispatch["config"]
    assert aliased_result is aliased_output
    assert torch.equal(result, unfused_result)
    assert torch.equal(aliased_result, unfused_result)
    assert not torch.allclose(reference, unclamped_reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result, reference, rtol=1e-1, atol=1e-2)
    torch.testing.assert_close(aliased_result, reference, rtol=1e-1, atol=1e-2)


@pytest.mark.fused_experts_impl
def test_fused_moe_rejects_mismatched_routing_batch():
    device = flag_gems.device
    hidden = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
    w1 = torch.randn(2, 16, 16, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(2, 16, 8, device=device, dtype=torch.bfloat16)
    topk_ids = torch.zeros(1, 1, device=device, dtype=torch.int32)
    topk_weights = torch.ones(1, 1, device=device, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="one row per input token"):
        flag_gems.fused_experts_impl(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
        )


@pytest.mark.fused_experts_impl
def test_pad_aware_clamped_swiglu_zeros_remote_routes():
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    device = flag_gems.device
    dtype = torch.bfloat16
    gate_up = 20 * torch.randn(4, 16, device=device, dtype=dtype)
    output = torch.full((4, 8), torch.nan, device=device, dtype=dtype)
    topk_ids = torch.tensor([[0, 2], [-1, 1]], device=device, dtype=torch.int32)
    expert_map = torch.tensor([0, 1, -1, -1], device=device, dtype=torch.int32)

    fused_moe.apply_moe_activation(
        fused_moe.MoEActivation.SILU,
        output,
        gate_up,
        clamp_limit=10.0,
        topk_ids=topk_ids,
        expert_map=expert_map,
        num_local_experts=2,
    )
    torch_device_fn.synchronize()

    gate, up = gate_up.float().chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(
        min=-10.0, max=10.0
    )
    torch.testing.assert_close(output[0], expected[0].to(dtype))
    torch.testing.assert_close(output[3], expected[3].to(dtype))
    assert torch.count_nonzero(output[1:3]).item() == 0


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
