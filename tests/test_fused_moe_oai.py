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
    ]
    + [
        ({name: value}, name)
        for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")
        for value in (-1e39, 1e39)
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


def test_oai_params_accept_fp32_finite_boundaries():
    limit = torch.finfo(torch.float32).max
    assert moe._validate_swigluoai_uninterleave_params(limit, -limit, limit) == (
        limit,
        -limit,
        limit,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_oai_clamp_preserves_nan(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.tensor(
        [
            [float("nan"), 1.0],
            [1.0, float("nan")],
            [float("nan"), float("nan")],
            [float("inf"), 1.0],
            [1.0, -float("inf")],
            [0.0, 1.0],
        ],
        device="cuda",
        dtype=dtype,
    )
    out = torch.empty((x.size(0), 1), device=x.device, dtype=dtype)
    moe.apply_moe_activation(
        moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        out,
        x,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    torch.testing.assert_close(out, _reference(x, 1.702, 1.0, 7.0), equal_nan=True)


@pytest.mark.parametrize(
    "large_stride", ["input_row", "input_column", "output_row", "output_column"]
)
def test_oai_address_offsets_exceed_int32(large_stride):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Touch only a few logical elements, but cross 2**31 elements in each
    # address expression. The backing allocation needs slightly over 4 GiB.
    if torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        pytest.skip("large-stride regression requires 5 GiB free device memory")
    row_stride = 1 << 30
    column_stride = (1 << 31) // 3 + 1
    if large_stride.endswith("row"):
        rows, n_inter = 3, 1
    else:
        rows, n_inter = 1, 2 if large_stride.startswith("input") else 3
    input_strides = (2 * n_inter, 1)
    output_strides = (n_inter, 1)
    if large_stride == "input_row":
        input_strides = (row_stride, 1)
    elif large_stride == "input_column":
        input_strides = (1, column_stride)
    elif large_stride == "output_row":
        output_strides = (row_stride, 1)
    else:
        output_strides = (1, row_stride)
    x = torch.empty_strided(
        (rows, 2 * n_inter), input_strides, device="cuda", dtype=torch.float16
    )
    out = torch.empty_strided(
        (rows, n_inter), output_strides, device=x.device, dtype=x.dtype
    )
    x.copy_(
        torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype).view(x.shape)
    )
    moe.apply_moe_activation(
        moe.MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        out,
        x,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    torch.testing.assert_close(out, _reference(x, 1.702, 1.0, 7.0))


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


def literal_quant(x):
    scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
    return (x.float() / scale).round().clamp(-128, 127).to(torch.int8), scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,i", [(1, 384), (7, 768), (32, 1536)])
@pytest.mark.parametrize("alpha,beta,limit", [(1.702, 1.0, 7.0), (0.5, -0.25, 3.0)])
def test_w8a8_oai_experts(dtype, m, i, alpha, beta, limit):
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
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    if m == 1:
        kwargs = dict(
            activation="swigluoai_uninterleave",
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=s1,
            w2_scale=s2,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=limit,
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
                activated = _reference(up.to(dtype), alpha, beta, limit)
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("alpha,beta,limit", [(1.702, 1.0, 7.0), (0.5, -0.25, 3.0)])
def test_unquantized_oai_experts(dtype, alpha, beta, limit):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(47)
    m, e, h, i, k = 7, 4, 128, 384, 2
    x = torch.randn((m, h), device="cuda", dtype=dtype)
    w1 = torch.randn((e, 2 * i, h), device="cuda", dtype=dtype) * 0.1
    w2 = torch.randn((e, h, i), device="cuda", dtype=dtype) * 0.1
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
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
    )
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        contributions = []
        for t in range(m):
            row = []
            for rank in range(k):
                expert = int(ids[t, rank])
                up = (x[t : t + 1].float() @ w1[expert].float().t()).to(dtype)
                activated = _reference(up, alpha, beta, limit)
                down = activated.float() @ w2[expert].float().t()
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
