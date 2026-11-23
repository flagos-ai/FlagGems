# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="NVIDIA fused MoE reduction regression"
)


def _inputs(rows, experts, hidden, intermediate, topk, dtype):
    torch.manual_seed(20260920)
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    w1 = torch.randn(experts, 2 * intermediate, hidden, device="cuda", dtype=dtype)
    w2 = torch.randn(experts, hidden, intermediate, device="cuda", dtype=dtype)
    w1 *= hidden**-0.5
    w2 *= intermediate**-0.5
    logits = torch.randn(rows, experts, device="cuda")
    weights, ids = logits.topk(topk, dim=-1)
    return x, w1, w2, weights.softmax(dim=-1), ids.to(torch.int32)


@pytest.mark.parametrize("rows", [1024, 4095, 4096, 4097])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_repeat_and_graph_reuse(rows, dtype):
    # Qwen3.8-Flash TP8 geometry. The previous atomic reduction first became
    # active at 4096 tokens; test both sides without changing kernel policy.
    args = _inputs(rows, 512, 2560, 80, 10, dtype)
    expected = flag_gems.fused_experts_impl(*args).clone()
    for _ in range(8):
        assert torch.equal(flag_gems.fused_experts_impl(*args), expected)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        flag_gems.fused_experts_impl(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = flag_gems.fused_experts_impl(*args)
    original = args[0].clone()
    for scale in (0.5, 1.0, -1.0, 1.0):
        args[0].copy_(original * scale)
        expected = flag_gems.fused_experts_impl(*args)
        graph.replay()
        assert torch.equal(output, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("topk", [1, 4])
def test_fused_moe_reduction_reference(dtype, topk):
    x, w1, w2, weights, ids = _inputs(4096, 8, 128, 64, topk, dtype)
    result = flag_gems.fused_experts_impl(x, w1, w2, weights, ids)
    routed = torch.empty(x.shape[0], topk, x.shape[1], device=x.device, dtype=dtype)
    # Independent grouped PyTorch reference. The existing plain-half GEMM1
    # path rounds gate, sigmoid and up before its two half-precision products.
    # Each weighted expert output is rounded once, then reduced in FP32.
    for expert in range(w1.shape[0]):
        tokens, slots = torch.where(ids == expert)
        gate, up = (x[tokens].float() @ w1[expert].float().T).chunk(2, dim=-1)
        activated = gate.to(dtype) * gate.sigmoid().to(dtype) * up.to(dtype)
        out = activated.float() @ w2[expert].float().T
        routed[tokens, slots] = (out * weights[tokens, slots, None]).to(dtype)
    expected = routed.float().sum(dim=1).to(dtype)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)
