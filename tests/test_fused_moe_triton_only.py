# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import flag_gems

from .test_fused_experts_impl import torch_fused_moe_ep_reference


class _AllocationAndViewsOnly(TorchDispatchMode):
    """Permit PyTorch metadata/allocation, but no ATen compute in the hot path."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        allowed = {
            "aten::empty",
            "aten::empty_like",
            "aten::empty_strided",
            "aten::view",
            "aten::_unsafe_view",
            "aten::slice",
            "aten::as_strided",
            "aten::alias",
            "aten::detach",
        }
        assert func._schema.name in allowed, f"Non-Triton compute: {func}"
        return func(*args, **(kwargs or {}))


def _reject_tle(*args, **kwargs):
    raise AssertionError("TLE dispatch must not be selected")


@pytest.mark.moe_align_block_size
@pytest.mark.parametrize("num_experts", [32, 288])
@pytest.mark.parametrize("map_dtype", [torch.int32, torch.int64])
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="CUDA routing fallback coverage"
)
def test_alignment_standard_triton_fallback(monkeypatch, num_experts, map_dtype):
    module = importlib.import_module("flag_gems.fused.moe_align_block_size")
    torch.manual_seed(20260906)
    topk_ids = torch.randint(
        num_experts, (512, 8), device=flag_gems.device, dtype=torch.int64
    )
    expert_map = torch.arange(
        num_experts - 1, -1, -1, device=flag_gems.device, dtype=map_dtype
    )
    expert_map[1::2] = -1
    # A live last map entry exercises the old negative-index padding behavior.
    expert_map[-1] = 2**40 if map_dtype == torch.int64 else 7
    ids_cpu = topk_ids.cpu().flatten().tolist()
    map_cpu = expert_map.cpu().tolist()
    block_size = 64
    monkeypatch.setattr(module, "HAS_TLE", True)
    monkeypatch.setattr(module, "_pick_tle_atomic_fused_launch_params", _reject_tle)

    # Standalone callers retain the original default TLE selection policy.
    with pytest.raises(AssertionError, match="TLE dispatch"):
        module.moe_align_block_size(topk_ids, block_size, num_experts, expert_map)

    with _AllocationAndViewsOnly():
        actual = module.moe_align_block_size(
            topk_ids, block_size, num_experts, expert_map, allow_tle=False
        )
    sorted_ids, mapped_ids, total = [x.cpu() for x in actual]
    assert mapped_ids.dtype == map_dtype
    offset = 0
    for expert in range(num_experts):
        expected = [route for route, value in enumerate(ids_cpu) if value == expert]
        padded_count = (len(expected) + block_size - 1) // block_size * block_size
        observed = sorted_ids[offset : offset + padded_count].tolist()
        assert sorted(route for route in observed if route < len(ids_cpu)) == expected
        assert all(route == len(ids_cpu) for route in observed if route >= len(ids_cpu))
        assert mapped_ids[
            offset // block_size : (offset + padded_count) // block_size
        ].tolist() == [map_cpu[expert]] * (padded_count // block_size)
        offset += padded_count
    assert total.item() == offset
    assert (mapped_ids[offset // block_size :] == map_cpu[-1]).all()


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("num_tokens", [512, 2048])
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="real EP prefill shape requires NVIDIA SM90",
)
def test_fused_moe_ep_prefill_standard_triton(monkeypatch, num_tokens, record_property):
    """Check real BF16 prefill, caller-buffer aliasing and dynamic graph replay."""
    module = importlib.import_module("flag_gems.fused.moe_align_block_size")
    fused_moe = importlib.import_module("flag_gems.fused.fused_moe")
    monkeypatch.setattr(module, "HAS_TLE", True)
    monkeypatch.setattr(module, "_pick_tle_atomic_fused_launch_params", _reject_tle)
    original_align = fused_moe.moe_align_block_size
    align_calls = 0

    def checked_align(*args, **kwargs):
        nonlocal align_calls
        align_calls += 1
        assert kwargs["allow_tle"] is False
        return original_align(*args, **kwargs)

    monkeypatch.setattr(fused_moe, "moe_align_block_size", checked_align)
    torch.manual_seed(20260906 + num_tokens)
    m, h, i, local_e, global_e, topk = num_tokens, 4096, 2048, 18, 288, 8
    kw = {"device": flag_gems.device, "dtype": torch.bfloat16}
    hidden = 4 * torch.randn((m, h), **kw)
    w1 = torch.randn((local_e, 2 * i, h), **kw) * h**-0.5
    w2 = torch.randn((local_e, h, i), **kw) * i**-0.5
    weights = torch.rand((m, topk), device=flag_gems.device, dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    ids = torch.rand((m, global_e), device=flag_gems.device).topk(topk, dim=-1).indices
    expert_map = torch.full((global_e,), -1, device=flag_gems.device, dtype=torch.int64)
    shard_start = 7 * local_e
    expert_map[shard_start : shard_start + local_e] = torch.arange(
        local_e - 1, -1, -1, device=flag_gems.device, dtype=torch.int64
    )
    cache13 = torch.empty(m * topk * max(2 * i, h), **kw)
    cache2 = torch.empty(m * topk * i, **kw)
    output = cache2[: m * h].view(m, h)

    def run():
        return flag_gems.fused_experts_impl(
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
        )

    # Warm up compilation/autotuning outside the operator-compute guard.
    run()
    with _AllocationAndViewsOnly():
        assert run() is output
    actual = output.clone()
    reference = torch_fused_moe_ep_reference(
        hidden, w1, w2, weights, ids, expert_map, clamp_limit=10.0
    )
    torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.03)
    relative_l2 = torch.linalg.vector_norm(
        actual.float() - reference.float()
    ) / torch.linalg.vector_norm(reference.float())
    assert relative_l2 < 0.004
    record_property("relative_l2", relative_l2.item())
    record_property("max_abs_error", (actual - reference).abs().max().item())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), _AllocationAndViewsOnly():
        run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, actual, rtol=0, atol=0)
    # Reusing the graph must observe a changed EP map and clear remote rows.
    expert_map.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    assert align_calls == 4


@pytest.mark.fused_experts_impl
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="CUDA direct-sum boundary coverage"
)
def test_fused_moe_direct_sum_uses_triton_fill(monkeypatch):
    """Small expert dimensions isolate the 4096-token direct-sum boundary."""
    module = importlib.import_module("flag_gems.fused.fused_moe")
    align = importlib.import_module("flag_gems.fused.moe_align_block_size")
    monkeypatch.setattr(align, "HAS_TLE", True)
    monkeypatch.setattr(align, "_pick_tle_atomic_fused_launch_params", _reject_tle)
    fill = module.fill_scalar_
    calls = 0

    def checked_fill(tensor, value):
        nonlocal calls
        calls += 1
        assert tensor.shape == (4096, 1, 128)
        assert value == 0
        return fill(tensor, value)

    monkeypatch.setattr(module, "fill_scalar_", checked_fill)
    torch.manual_seed(20260906)
    kw = {"device": flag_gems.device, "dtype": torch.bfloat16}
    hidden = torch.randn((4096, 128), **kw)
    w1 = torch.randn((8, 128, 128), **kw) * 128**-0.5
    w2 = torch.randn((8, 128, 64), **kw) * 64**-0.5
    ids = torch.rand((4096, 8), device=flag_gems.device).topk(2, dim=-1).indices
    weights = torch.full((4096, 2), 0.5, device=flag_gems.device)
    args = hidden, w1, w2, weights, ids
    module.fused_experts_impl(*args)
    calls = 0
    with _AllocationAndViewsOnly():
        actual = module.fused_experts_impl(*args)
    assert calls == 1
    # Compare to the regular GEMM2 + top-k reduction, without changing GEMM1.
    monkeypatch.setattr(module, "MOE_DIRECT_SUM_MIN_TOKENS", 8192)
    expected = module.fused_experts_impl(*args)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)
