# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Focused numerical checks for the SGLang DeepSeek V4.1 operator port."""

import pytest
import torch

from flag_gems.fused.dsv41.block_fp8_linear import block_fp8_linear
from flag_gems.fused.dsv41.fp4_indexer import (
    fp4_index_logits_decode,
    fp4_index_logits_req_to_token,
)
from flag_gems.fused.dsv41.mhc import hc_combine, hc_mix_stats


def _e2m1_decode(packed):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device
    )
    value = magnitudes[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -value, value)


def test_block_fp8_rejects_encoded_scales():
    x = torch.empty((1, 32), dtype=torch.bfloat16)
    w = torch.empty((32, 32), dtype=torch.uint8)
    with pytest.raises(TypeError, match="numeric scales"):
        block_fp8_linear(x, w, (32, 32), torch.ones((1, 1), dtype=torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_block_fp8_linear_modes_match_independent_reference():
    torch.manual_seed(413)
    device = "cuda"
    m, n, k = 3, 35, 128
    x = (torch.randn(m, k, device=device) * 0.5).bfloat16()
    w = (torch.randn(n, k, device=device) * 0.5).to(torch.float8_e4m3fn)
    ws = torch.tensor(
        [[0.03125, 0.5, 2.0, 16.0], [8.0, 1.0, 0.25, 0.0625]], device=device
    )

    actual_bf16 = block_fp8_linear(x, w, (32, 32), ws, mode="bf16")
    expected_bf16 = torch.zeros((m, n), device=device)
    for group in range(k // 32):
        start = group * 32
        scale = ws[:, group].repeat_interleave(32)[:n]
        weight_tile = (w[:, start : start + 32].float() * scale[:, None]).bfloat16()
        expected_bf16 += x[:, start : start + 32].float() @ weight_tile.float().T
    torch.testing.assert_close(actual_bf16.float(), expected_bf16, rtol=0.02, atol=0.25)

    if torch.cuda.get_device_capability() < (8, 9):
        return
    q = x.to(torch.float8_e4m3fn)
    act_scales = torch.tensor(
        [[0.5, 0.25, 0.125, 1.0], [0.25, 0.5, 1.0, 2.0], [1.0, 2.0, 0.5, 0.25]],
        device=device,
    )
    actual_fp8 = block_fp8_linear(
        q, w, (32, 32), ws, input_scale=act_scales, native_fp8=True
    )
    expected_fp8 = torch.zeros((m, n), device=device)
    for group in range(k // 32):
        start = group * 32
        dot = q[:, start : start + 32].float() @ w[:, start : start + 32].float().T
        weight_scale = ws[:, group].repeat_interleave(32)[:n]
        expected_fp8 += dot * act_scales[:, group, None] * weight_scale[None, :]
    torch.testing.assert_close(actual_fp8.float(), expected_fp8, rtol=0.02, atol=0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_indexer_packed_pages_and_nan_scale():
    torch.manual_seed(517)
    b, h, length, page = 3, 32, 65, 64
    payload = torch.randint(0, 256, (128, 64), dtype=torch.uint8)
    scale = torch.randint(125, 129, (128, 4), dtype=torch.uint8)
    scale[37, 2] = 255  # Reserved UE8M0 NaN must propagate.
    table = torch.cat([payload.reshape(2, page * 64), scale.reshape(2, page * 4)], 1)
    decoded_scale = torch.ldexp(
        torch.ones_like(scale, dtype=torch.float32), scale.int() - 127
    )
    decoded_scale = torch.where(scale == 255, float("nan"), decoded_scale)
    keys = (_e2m1_decode(payload) * decoded_scale.repeat_interleave(32, 1)).bfloat16()
    q = torch.randn(b, h, 128).bfloat16()
    weights = torch.randn(b, h).bfloat16()
    slots = torch.stack(
        [torch.arange(length), torch.arange(length) + 31, torch.arange(length) + 15]
    )
    lens = torch.tensor([0, 63, 65])
    expected = []
    for row in range(b):
        dot = (q[row].float() @ keys[slots[row]].float().T).bfloat16()
        scores = (
            (dot.relu().float() * weights[row, :, None].float())
            .bfloat16()
            .float()
            .sum(0)
            .bfloat16()
            .float()
        )
        scores[lens[row] :] = -torch.inf
        expected.append(scores)
    actual = fp4_index_logits_decode(
        q.cuda(), weights.cuda(), slots.cuda(), lens.cuda(), table.cuda(), page
    )
    torch.testing.assert_close(
        actual.cpu(), torch.stack(expected), rtol=0.008, atol=0.25, equal_nan=True
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp4_indexer_grouped_six_queries_match_fallback():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("grouped path is selected only on SM90")
    torch.manual_seed(521)
    requests, queries_per_request, width, page = 20, 6, 64, 64
    b = requests * queries_per_request
    q = torch.randn(b, 32, 128, device="cuda").bfloat16()
    weights = torch.randn(b, 32, device="cuda").bfloat16()
    req = torch.arange(requests, device="cuda").repeat_interleave(queries_per_request)
    req_to_token = torch.arange(width, device="cuda", dtype=torch.int32).repeat(
        requests, 1
    )
    lens = torch.full((b,), width, device="cuda", dtype=torch.int64)
    table = torch.randint(
        0, 256, (1, page * 64 + page * 4), device="cuda", dtype=torch.uint8
    )
    table[:, page * 64 :] = 127  # Unit UE8M0 scale.
    baseline = fp4_index_logits_req_to_token(
        q, weights, req_to_token, req, lens, table, page, 1, width
    )
    grouped = fp4_index_logits_req_to_token(
        q,
        weights,
        req_to_token,
        req,
        lens,
        table,
        page,
        1,
        width,
        query_group_size=queries_per_request,
    )
    torch.testing.assert_close(grouped, baseline, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mhc_fp32_weights_and_predecessor_combine():
    torch.manual_seed(713)
    x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(24, 512, device="cuda", dtype=torch.float32) * 0.01
    expected = (x.float() @ weight.T) * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6
    )
    actual = hc_mix_stats(x, weight, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
    previous = torch.rand(7, 4, device="cuda")
    expected_y = (x.reshape(7, 4, 128).float() * previous[..., None]).sum(1).bfloat16()
    torch.testing.assert_close(
        hc_combine(x, previous, 4, x.dtype), expected_y, rtol=0.008, atol=0.01
    )
