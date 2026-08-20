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

import pytest
import torch
import torch.nn.functional as F
import triton

import flag_gems

chunk_gated_delta_rule_module = importlib.import_module(
    "flag_gems.fused.chunk_gated_delta_rule"
)

_TRITON_ALLOCATOR_READY = False


def _cuda_available() -> bool:
    return torch.cuda.is_available() and flag_gems.device == "cuda"


pytestmark = [
    pytest.mark.chunk_gated_delta_rule,
    pytest.mark.skipif(
        not _cuda_available(), reason="chunk gated delta rule tests require CUDA"
    ),
]


@pytest.fixture(autouse=True)
def _install_triton_allocator():
    global _TRITON_ALLOCATOR_READY
    if (
        _TRITON_ALLOCATOR_READY
        or not _cuda_available()
        or not hasattr(triton, "set_allocator")
    ):
        return

    def _alloc(size: int, _alignment: int, _stream: int | None):
        return torch.empty((size,), dtype=torch.uint8, device=flag_gems.device)

    triton.set_allocator(_alloc)
    _TRITON_ALLOCATOR_READY = True


def _seq_first(x: torch.Tensor, head_first: bool) -> torch.Tensor:
    return x.transpose(1, 2) if head_first else x


def _public_layout(x: torch.Tensor, head_first: bool) -> torch.Tensor:
    return x.transpose(1, 2) if head_first else x


def _stable_decay(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    device = flag_gems.device
    decay = (
        torch.empty(shape, device=device, dtype=torch.float32)
        .uniform_(-4.605170185988091, -3.506557897319982)
        .exp()
    )
    return torch.log1p(-decay).to(dtype)


def _stable_beta(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    device = flag_gems.device
    return (
        torch.empty(shape, device=device, dtype=torch.float32)
        .uniform_(-2.0, 2.0)
        .sigmoid()
        .to(dtype)
    )


def _strided_last_dim(data: torch.Tensor) -> torch.Tensor:
    expanded = torch.empty(
        *data.shape[:-1],
        data.shape[-1] * 2,
        device=data.device,
        dtype=data.dtype,
    )
    expanded[..., ::2].copy_(data)
    return expanded[..., ::2]


def _make_inputs(
    *,
    B: int,
    T: int,
    Hg: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    head_first: bool,
    non_contiguous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = flag_gems.device
    q_data = torch.randn(B, T, Hg, K, device=device, dtype=torch.float32).to(dtype)
    k_data = F.normalize(
        torch.randn(B, T, Hg, K, device=device, dtype=torch.float32),
        p=2.0,
        dim=-1,
        eps=1e-6,
    ).to(dtype)
    v_data = (0.125 * torch.randn(B, T, H, V, device=device, dtype=torch.float32)).to(
        dtype
    )
    if non_contiguous:
        q = _strided_last_dim(q_data)
        k = _strided_last_dim(k_data)
        v = _strided_last_dim(v_data)
    else:
        q = q_data
        k = k_data
        v = v_data
    g = _stable_decay((B, T, H), dtype)
    beta = _stable_beta((B, T, H), dtype)
    return (
        _public_layout(q, head_first),
        _public_layout(k, head_first),
        _public_layout(v, head_first),
        _public_layout(beta, head_first),
        _public_layout(g, head_first),
    )


def _reference_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    head_first: bool,
    scale: float | None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q = _seq_first(q, head_first).float()
    k = _seq_first(k, head_first).float()
    v_seq = _seq_first(v, head_first)
    v_float = v_seq.float()
    beta = _seq_first(beta, head_first).float()
    g = _seq_first(g, head_first).float()
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2.0, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2.0, dim=-1, eps=1e-6)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, Hg, K = q.shape
    H, V = v_seq.shape[2], v_seq.shape[3]
    heads_per_group = H // Hg
    out = torch.empty_like(v_float)
    final_state = (
        torch.empty(
            B if cu_seqlens is None else cu_seqlens.numel() - 1,
            H,
            K,
            V,
            device=v.device,
            dtype=torch.float32,
        )
        if output_final_state
        else None
    )

    spans: list[tuple[int, int, int, int]]
    if cu_seqlens is None:
        spans = [(b, b, 0, T) for b in range(B)]
    else:
        cu_cpu = cu_seqlens.detach().cpu().tolist()
        spans = [(0, n, cu_cpu[n], cu_cpu[n + 1]) for n in range(len(cu_cpu) - 1)]

    for batch_idx, state_idx, start, end in spans:
        if initial_state is None:
            h = torch.zeros(H, K, V, device=v.device, dtype=torch.float32)
        else:
            h = initial_state[state_idx].float().clone()
        for t in range(start, end):
            q_t = q[batch_idx, t]
            k_t = k[batch_idx, t]
            for hv in range(H):
                hg = hv // heads_per_group
                h[hv] *= torch.exp(g[batch_idx, t, hv])
                kv = k_t[hg]
                residual = v_float[batch_idx, t, hv] - torch.matmul(kv, h[hv])
                u = residual * beta[batch_idx, t, hv]
                h[hv] += kv[:, None] * u[None, :]
                out[batch_idx, t, hv] = torch.matmul(q_t[hg] * scale, h[hv])
        if output_final_state:
            final_state[state_idx] = h

    out = out.to(v.dtype)
    if head_first:
        out = out.transpose(1, 2)
    return out, final_state


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
    *,
    final_state: bool = False,
) -> None:
    if dtype == torch.float32:
        atol, rtol = (2e-2, 2e-2) if final_state else (1e-2, 1e-2)
    else:
        atol, rtol = (3e-1, 3e-1) if final_state else (1.5e-1, 1.5e-1)
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=atol, rtol=rtol, check_dtype=False
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_first", [True, False])
def test_chunk_gated_delta_rule_matches_reference_without_final_state(
    dtype, head_first
):
    torch.manual_seed(1000 + int(head_first))
    q, k, v, beta, g = _make_inputs(
        B=1, T=64, Hg=2, H=4, K=64, V=32, dtype=dtype, head_first=head_first
    )

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q, k, v, beta, g, head_first=head_first, output_final_state=False
    )
    expected, expected_final = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        head_first=head_first,
        scale=None,
    )

    assert actual_final is None
    assert expected_final is None
    _assert_close(actual, expected, dtype)


def test_chunk_gated_delta_rule_uses_initial_state_and_returns_final_state():
    dtype = torch.float32
    torch.manual_seed(2000)
    q, k, v, beta, g = _make_inputs(
        B=2, T=33, Hg=2, H=4, K=64, V=32, dtype=dtype, head_first=False
    )
    initial_state = 0.125 * torch.randn(
        2, 4, 64, 32, device=flag_gems.device, dtype=dtype
    )

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        output_final_state=True,
        head_first=False,
    )
    expected, expected_final = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        head_first=False,
        scale=None,
    )

    _assert_close(actual, expected, dtype)
    _assert_close(actual_final, expected_final, dtype, final_state=True)


def test_chunk_gated_delta_rule_supports_two_sequence_varlen_pack():
    dtype = torch.float16
    torch.manual_seed(3000)
    q, k, v, beta, g = _make_inputs(
        B=1, T=80, Hg=2, H=4, K=64, V=32, dtype=dtype, head_first=False
    )
    cu_seqlens = torch.tensor([0, 17, 80], device=flag_gems.device, dtype=torch.long)

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        head_first=False,
    )
    expected, expected_final = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        scale=None,
    )

    _assert_close(actual, expected, dtype)
    _assert_close(actual_final, expected_final, dtype, final_state=True)


def test_chunk_gated_delta_rule_accepts_non_contiguous_seq_first_inputs():
    dtype = torch.float32
    torch.manual_seed(4000)
    q, k, v, beta, g = _make_inputs(
        B=1,
        T=64,
        Hg=2,
        H=4,
        K=64,
        V=32,
        dtype=dtype,
        head_first=False,
        non_contiguous=True,
    )
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    actual, _ = flag_gems.chunk_gated_delta_rule(
        q, k, v, beta, g, head_first=False, output_final_state=False
    )
    expected, _ = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        head_first=False,
        scale=None,
    )

    _assert_close(actual, expected, dtype)


def test_chunk_gated_delta_rule_supports_qk_l2norm_option():
    dtype = torch.float32
    torch.manual_seed(5000)
    q, k, v, beta, g = _make_inputs(
        B=1, T=32, Hg=2, H=4, K=64, V=32, dtype=dtype, head_first=True
    )

    actual, _ = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        head_first=True,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    expected, _ = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        head_first=True,
        scale=None,
        use_qk_l2norm_in_kernel=True,
    )

    _assert_close(actual, expected, dtype)


def test_chunk_gated_delta_rule_supports_qk_l2norm_on_chunk_path():
    dtype = torch.float32
    torch.manual_seed(6000)
    q, k, v, beta, g = _make_inputs(
        B=1, T=33, Hg=2, H=4, K=64, V=32, dtype=dtype, head_first=True
    )
    initial_state = 0.125 * torch.randn(
        1, 4, 64, 32, device=flag_gems.device, dtype=dtype
    )

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        head_first=True,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    expected, expected_final = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        head_first=True,
        scale=None,
        use_qk_l2norm_in_kernel=True,
    )

    _assert_close(actual, expected, dtype)
    _assert_close(actual_final, expected_final, dtype, final_state=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_gated_delta_rule_fused_tail_vblock_matches_reference(dtype):
    torch.manual_seed(7000 + (0 if dtype == torch.float16 else 1))
    q, k, v, beta, g = _make_inputs(
        B=1, T=128, Hg=4, H=8, K=64, V=64, dtype=dtype, head_first=False
    )
    initial_state = 0.125 * torch.randn(
        1, 8, 64, 64, device=flag_gems.device, dtype=dtype
    )

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        output_final_state=True,
        head_first=False,
    )
    expected, expected_final = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        head_first=False,
        scale=None,
    )

    _assert_close(actual, expected, dtype)
    _assert_close(actual_final, expected_final, dtype, final_state=True)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "B": 1,
                "T": 16,
                "Hg": 1,
                "H": 4,
                "K": 16,
                "V": 16,
                "dtype": torch.float32,
                "head_first": False,
                "output_final_state": True,
            },
            id="single_chunk_gqa_fp32",
        ),
        pytest.param(
            {
                "B": 1,
                "T": 37,
                "Hg": 2,
                "H": 4,
                "K": 24,
                "V": 17,
                "dtype": torch.float16,
                "head_first": True,
                "output_final_state": False,
            },
            id="non_aligned_kv_fp16",
        ),
        pytest.param(
            {
                "B": 1,
                "T": 9,
                "Hg": 2,
                "H": 8,
                "K": 32,
                "V": 16,
                "dtype": torch.bfloat16,
                "head_first": False,
                "output_final_state": True,
            },
            id="many_heads_bf16",
        ),
    ],
)
def test_chunk_gated_delta_rule_direct_path_shape_variants(case):
    torch.manual_seed(8000 + case["T"])
    input_case = dict(case)
    output_final_state = case["output_final_state"]
    input_case.pop("output_final_state")
    q, k, v, beta, g = _make_inputs(**input_case)

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        head_first=case["head_first"],
        output_final_state=output_final_state,
    )
    expected, expected_final = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=None,
        output_final_state=output_final_state,
        cu_seqlens=None,
        head_first=case["head_first"],
        scale=None,
    )

    _assert_close(actual, expected, case["dtype"])
    if output_final_state:
        _assert_close(actual_final, expected_final, case["dtype"], final_state=True)
    else:
        assert actual_final is None
        assert expected_final is None


def test_chunk_gated_delta_rule_varlen_initial_state_gqa_non_aligned_lengths():
    dtype = torch.float32
    torch.manual_seed(9000)
    q, k, v, beta, g = _make_inputs(
        B=1, T=70, Hg=1, H=4, K=32, V=16, dtype=dtype, head_first=False
    )
    cu_seqlens = torch.tensor(
        [0, 13, 37, 70], device=flag_gems.device, dtype=torch.long
    )
    initial_state = 0.125 * torch.randn(
        3, 4, 32, 16, device=flag_gems.device, dtype=dtype
    )

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        head_first=False,
    )
    expected, expected_final = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        scale=None,
    )

    _assert_close(actual, expected, dtype)
    _assert_close(actual_final, expected_final, dtype, final_state=True)


def test_chunk_gated_delta_rule_zero_values_extreme_gates_return_zero_state():
    dtype = torch.float32
    torch.manual_seed(10000)
    q, k, v, beta, g = _make_inputs(
        B=1, T=17, Hg=1, H=2, K=16, V=16, dtype=dtype, head_first=False
    )
    v.zero_()
    beta.fill_(1)
    g.fill_(-20)

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        output_final_state=True,
        head_first=False,
    )

    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
    torch.testing.assert_close(
        actual_final,
        torch.zeros_like(actual_final),
        atol=0,
        rtol=0,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "case",
    [
        "zero_query",
        "zero_key",
        "zero_value",
        "beta_zero",
        "beta_one",
        "weak_decay",
        "strong_decay",
        "custom_scale",
    ],
)
def test_chunk_gated_delta_rule_value_edge_cases_match_reference(case):
    dtype = torch.float32
    torch.manual_seed(11000)
    q, k, v, beta, g = _make_inputs(
        B=1, T=8, Hg=1, H=2, K=16, V=16, dtype=dtype, head_first=False
    )
    scale = None

    if case == "zero_query":
        q.zero_()
    elif case == "zero_key":
        k.zero_()
    elif case == "zero_value":
        v.zero_()
    elif case == "beta_zero":
        beta.zero_()
    elif case == "beta_one":
        beta.fill_(1)
    elif case == "weak_decay":
        g.fill_(-1e-4)
    elif case == "strong_decay":
        g.fill_(-20)
    elif case == "custom_scale":
        scale = 0.25

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        output_final_state=True,
        head_first=False,
        scale=scale,
    )
    expected, expected_final = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=None,
        head_first=False,
        scale=scale,
    )

    _assert_close(actual, expected, dtype)
    _assert_close(actual_final, expected_final, dtype, final_state=True)


def test_chunk_gated_delta_rule_accepts_non_contiguous_head_first_inputs():
    dtype = torch.float32
    torch.manual_seed(12000)
    q, k, v, beta, g = _make_inputs(
        B=1,
        T=32,
        Hg=2,
        H=4,
        K=32,
        V=16,
        dtype=dtype,
        head_first=True,
        non_contiguous=True,
    )
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        head_first=True,
        output_final_state=False,
    )
    expected, expected_final = _reference_chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        head_first=True,
        scale=None,
    )

    assert actual_final is None
    assert expected_final is None
    _assert_close(actual, expected, dtype)


def test_chunk_gated_delta_rule_repeated_calls_are_deterministic():
    dtype = torch.float32
    torch.manual_seed(13000)
    q, k, v, beta, g = _make_inputs(
        B=1, T=32, Hg=2, H=4, K=32, V=16, dtype=dtype, head_first=False
    )

    actual_1, final_1 = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        head_first=False,
        output_final_state=True,
    )
    actual_2, final_2 = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        head_first=False,
        output_final_state=True,
    )

    torch.testing.assert_close(actual_1, actual_2, atol=0, rtol=0)
    torch.testing.assert_close(final_1, final_2, atol=0, rtol=0)


@pytest.mark.parametrize(
    "case, match",
    [
        ("bt", "currently supports only BT=64"),
        ("head_ratio", "head count must divide"),
        ("cu_seqlens_dtype", "cu_seqlens must have dtype torch.long"),
        ("initial_state_shape", "initial_state must have shape"),
    ],
)
def test_chunk_gated_delta_rule_rejects_invalid_public_arguments(case, match):
    dtype = torch.float32
    q, k, v, beta, g = _make_inputs(
        B=1, T=16, Hg=2, H=4, K=16, V=16, dtype=dtype, head_first=False
    )
    kwargs = {"head_first": False}

    if case == "bt":
        kwargs["BT"] = 32
    elif case == "head_ratio":
        q, k, v, beta, g = _make_inputs(
            B=1, T=16, Hg=3, H=4, K=16, V=16, dtype=dtype, head_first=False
        )
    elif case == "cu_seqlens_dtype":
        kwargs["cu_seqlens"] = torch.tensor(
            [0, 16], device=flag_gems.device, dtype=torch.int32
        )
    elif case == "initial_state_shape":
        kwargs["initial_state"] = torch.zeros(
            1, 4, 16, 15, device=flag_gems.device, dtype=dtype
        )

    with pytest.raises(ValueError, match=match):
        flag_gems.chunk_gated_delta_rule(q, k, v, beta, g, **kwargs)


def test_chunk_gated_delta_rule_does_not_broadly_reject_iluvatar_chunk_path(
    monkeypatch,
):
    dtype = torch.float32
    torch.manual_seed(7000)
    q, k, v, beta, g = _make_inputs(
        B=1, T=129, Hg=2, H=4, K=64, V=32, dtype=dtype, head_first=False
    )
    calls = []

    def _fake_chunk_fwd(**kwargs):
        calls.append(kwargs)
        return None, kwargs["v"].clone(), None, None, None, None, None

    monkeypatch.setattr(
        chunk_gated_delta_rule_module, "chunk_gated_delta_rule_fwd", _fake_chunk_fwd
    )

    actual, actual_final = flag_gems.chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        head_first=False,
        output_final_state=False,
    )

    assert calls
    assert actual_final is None
    torch.testing.assert_close(actual, v)
