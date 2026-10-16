"""Focused semantic checks for the FlagGems M3 MoE activation."""

from __future__ import annotations

import importlib

import pytest
import torch

moe = importlib.import_module("flag_gems.fused.fused_moe")
pytestmark = pytest.mark.fused_experts_impl


def _reference(gate_up: torch.Tensor, alpha: float, beta: float, limit: float):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float().clamp(max=limit)
    up = up.float().clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(gate_up.dtype)


def test_new_enum_is_distinct_from_legacy_alias():
    assert moe.MoEActivation.from_str("swigluoai_uninterleave") is (
        moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE
    )
    assert moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE is not (moe.MoEActivation.SWIGLUOAI)
    assert (
        moe.MoEActivation.adjust_N_for_activation(
            14, moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE
        )
        == 7
    )


@pytest.mark.parametrize(
    ("overrides", "name"),
    [
        ({"gemm1_alpha": None}, "gemm1_alpha"),
        ({"gemm1_beta": None}, "gemm1_beta"),
        ({"gemm1_clamp_limit": None}, "gemm1_clamp_limit"),
        ({"gemm1_alpha": float("nan")}, "gemm1_alpha"),
        ({"gemm1_beta": float("inf")}, "gemm1_beta"),
        ({"gemm1_clamp_limit": -float("inf")}, "gemm1_clamp_limit"),
        ({"gemm1_clamp_limit": -1.0}, "gemm1_clamp_limit"),
        ({"gemm1_alpha": torch.tensor(1.702)}, "gemm1_alpha"),
        ({"gemm1_alpha": 10**1000}, "gemm1_alpha"),
    ],
)
def test_invalid_oai_params_fail_before_tensor_work(overrides, name):
    params = dict(gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0)
    params.update(overrides)
    with pytest.raises(ValueError, match=name):
        moe.fused_experts_impl(
            None, None, None, None, None, activation="swigluoai_uninterleave", **params
        )


def test_legacy_alias_stays_unsupported_and_silu_rejects_oai_params():
    with pytest.raises(ValueError, match="Only 'silu'"):
        moe.fused_experts_impl(None, None, None, None, None, activation="swigluoai")
    with pytest.raises(ValueError, match="require swigluoai_uninterleave"):
        moe.fused_experts_impl(
            None, None, None, None, None, activation="silu", gemm1_alpha=1.702
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_oai_kernel_split_layout_clamp_stride_and_input_mutation(dtype):
    if not torch.cuda.is_available():
        pytest.skip("Triton device check runs in the pinned GPU image")

    device = torch.device("cuda")
    rows, n_inter = 3, 7  # tail mask and non-power-of-two output width
    alpha, beta, limit = 1.702, 1.0, 7.0
    input_storage = torch.empty((rows, 4 * n_inter), device=device, dtype=dtype)
    gate_up = input_storage[:, ::2]  # stride(1) == 2
    output_storage = torch.full((rows, 2 * n_inter), -999.0, device=device, dtype=dtype)
    output = output_storage[:, ::2]  # stride(1) == 2
    gate_up.copy_(
        torch.tensor(
            [
                [-10, -7, -1, 0, 1, 7, 10, -10, -7, -1, 0, 1, 7, 10],
                [10, 8, 5, 2, -2, -8, -10, 10, 8, 5, 2, -2, -8, -10],
                [3, 4, 5, 6, 7, 8, 9, -9, -8, -7, -6, -5, -4, -3],
            ],
            device=device,
            dtype=dtype,
        )
    )

    def launch():
        return moe.apply_moe_activation(
            moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            output,
            gate_up,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
        )

    assert launch() is output
    torch.testing.assert_close(output, _reference(gate_up, alpha, beta, limit))
    assert torch.all(output_storage[:, 1::2] == -999.0)

    # Warmed-up launch is capturable; a replay must read the mutated input.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    gate_up.copy_(torch.flip(gate_up, dims=(1,)))
    graph.replay()
    torch.testing.assert_close(output, _reference(gate_up, alpha, beta, limit))


@pytest.mark.parametrize("shape", [(0, 14), (3, 0)])
def test_oai_empty_output(shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    out = torch.empty((shape[0], shape[1] // 2), device="cuda", dtype=x.dtype)
    assert (
        moe.apply_moe_activation(
            moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            out,
            x,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
        )
        is out
    )


def literal_oai(x):
    gate, up = x.float().chunk(2, dim=-1)
    gate = gate.clamp(max=7.0)
    up = up.clamp(-7.0, 7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(x.dtype)


def literal_quant(x):
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,i", [(1, 384), (7, 768), (32, 1536)])
def test_w8a8_oai_experts(dtype, m, i):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(47)
    e, h, k = 4, 128, 2
    x = torch.randn((m, h), device="cuda", dtype=dtype)
    w1 = torch.randint(-64, 64, (e, 2 * i, h), device="cuda", dtype=torch.int8)
    w2 = torch.randint(-64, 64, (e, h, i), device="cuda", dtype=torch.int8)
    s1 = torch.rand((e, 2 * i), device="cuda") * 0.001 + 0.001
    s2 = torch.rand((e, h), device="cuda") * 0.001 + 0.001
    ids = (torch.arange(m * k, device="cuda").view(m, k) % e).int()
    weights = torch.rand((m, k), device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    actual = moe.fused_experts_impl(
        x,
        w1,
        w2,
        weights,
        ids,
        activation="swigluoai_uninterleave",
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=s1,
        w2_scale=s2,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    if m == 1:
        kwargs = dict(
            activation="swigluoai_uninterleave",
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=s1,
            w2_scale=s2,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
        )
        out = moe.outplace_fused_experts(x, w1, w2, weights, ids, **kwargs)
        torch.testing.assert_close(out, actual, rtol=0, atol=0)
        in_place = x.clone()
        assert (
            moe.inplace_fused_experts(in_place, w1, w2, weights, ids, **kwargs) is None
        )
        torch.testing.assert_close(in_place, actual, rtol=0, atol=0)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        qx, sx = literal_quant(x)
        contributions = []
        for t in range(m):
            row = []
            for rank in range(k):
                expert = int(ids[t, rank])
                up = (
                    (qx[t : t + 1].float() @ w1[expert].float().t())
                    * sx[t]
                    * s1[expert]
                )
                activated = literal_oai(up.to(dtype))
                qa, sa = literal_quant(activated)
                down = (qa.float() @ w2[expert].float().t()) * sa * s2[expert]
                row.append((down * weights[t, rank]).to(dtype))
            contributions.append(torch.stack(row).float().sum(0).to(dtype))
        expected = torch.cat(contributions)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.0005 if dtype == torch.float16 else 0.004,
        rtol=0.003 if dtype == torch.float16 else 0.016,
    )
