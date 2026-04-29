import pytest
import torch
import torch.nn.functional as F

import flag_gems

CUDA_AVAILABLE = torch.cuda.is_available() and flag_gems.device == "cuda"

DTYPE_TOLERANCES = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-4, 0.016),
}
DTYPES = [torch.float16, torch.bfloat16]


def _build_model_like_inputs(
    T: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    Hg: int = 4,
    H: int = 8,
    K: int = 64,
    V: int = 32,
):
    device = flag_gems.device
    q = torch.rand((1, Hg, T, K), device=device, dtype=torch.float32).to(dtype)
    k = torch.randn((1, Hg, T, K), device=device, dtype=torch.float32)
    k = F.normalize(k, dim=-1, p=2).to(dtype)
    v = torch.randn((1, H, T, V), device=device, dtype=torch.float32).to(dtype)
    beta = (
        torch.randn((1, H, T), device=device, dtype=torch.float32).sigmoid().to(dtype)
    )
    g = (
        torch.empty((1, H, T), device=device, dtype=torch.float32)
        .uniform_(0.01, 0.03)
        .log()
        .to(dtype)
    )
    initial_state = torch.zeros((1, H, K, V), device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)
    ssm_state_indices = torch.zeros(T, device=device, dtype=torch.long)
    scale = K**-0.5
    return {
        "q_head_first": q,
        "k_head_first": k,
        "v_head_first": v,
        "beta_head_first": beta,
        "g_head_first": g,
        "q": q.transpose(1, 2).contiguous(),
        "k": k.transpose(1, 2).contiguous(),
        "v": v.transpose(1, 2).contiguous(),
        "beta": beta.transpose(1, 2).contiguous(),
        "g": g.transpose(1, 2).contiguous(),
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "scale": float(scale),
    }


def _recurrent_gated_delta_rule_ref(q, k, v, beta, g):
    q, k, v, beta, g = [x.float() for x in (q, k, v, beta, g)]
    b, hq, l, d_k = q.shape
    hv = v.shape[1]
    group = hv // hq
    d_v = v.shape[-1]
    o = torch.zeros((b, hv, l, d_v), device=q.device, dtype=torch.float32)
    state = torch.zeros((b, hv, d_k, d_v), device=q.device, dtype=torch.float32)
    q = q * (d_k**-0.5)

    for i in range(l):
        for h in range(hv):
            kh = h // group
            cur_k = k[:, kh, i]
            cur_q = q[:, kh, i]
            cur_v = v[:, h, i].clone()
            state[:, h] = state[:, h] * g[:, h, i].exp()[:, None, None]
            cur_v = cur_v - (state[:, h] * cur_k[..., None]).sum(-2)
            cur_v = cur_v * beta[:, h, i][..., None]
            state[:, h] = state[:, h] + cur_k.unsqueeze(-1) * cur_v.unsqueeze(-2)
            o[:, h, i] = torch.einsum("bd,bdm->bm", cur_q, state[:, h])
    return o, state


def _run_chunk(inputs, *, output_final_state: bool, head_first: bool):
    if head_first:
        q = inputs["q_head_first"]
        k = inputs["k_head_first"]
        v = inputs["v_head_first"]
        beta = inputs["beta_head_first"]
        g = inputs["g_head_first"]
    else:
        q = inputs["q"]
        k = inputs["k"]
        v = inputs["v"]
        beta = inputs["beta"]
        g = inputs["g"]

    return flag_gems.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        BT=64,
        initial_state=inputs["initial_state"].clone(),
        output_final_state=output_final_state,
        cu_seqlens=inputs["cu_seqlens"],
        head_first=head_first,
        scale=inputs["scale"],
    )


def _run_recurrent(inputs):
    return flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=inputs["initial_state"].clone(),
        inplace_final_state=True,
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype):
    rtol, atol = DTYPE_TOLERANCES[dtype]
    actual = actual.to(dtype)
    expected = expected.to(dtype)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", [64, 128, 512])
def test_chunk_gated_delta_rule_matches_recurrent_ref(T, dtype):
    inputs = _build_model_like_inputs(T, dtype=dtype)
    ref_out, ref_final = _recurrent_gated_delta_rule_ref(
        inputs["q_head_first"],
        inputs["k_head_first"],
        inputs["v_head_first"],
        inputs["beta_head_first"],
        inputs["g_head_first"],
    )
    chunk_out, chunk_final = _run_chunk(
        inputs, output_final_state=True, head_first=True
    )

    _assert_close(chunk_out, ref_out, dtype)
    _assert_close(chunk_final, ref_final, dtype)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", [64, 128, 512])
def test_library_recurrent_matches_recurrent_ref(T, dtype):
    inputs = _build_model_like_inputs(T, dtype=dtype)
    ref_out, ref_final = _recurrent_gated_delta_rule_ref(
        inputs["q_head_first"],
        inputs["k_head_first"],
        inputs["v_head_first"],
        inputs["beta_head_first"],
        inputs["g_head_first"],
    )
    recurrent_out, recurrent_final = _run_recurrent(inputs)

    _assert_close(recurrent_out.transpose(1, 2), ref_out, dtype)
    _assert_close(recurrent_final, ref_final, dtype)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", [64, 128, 512])
def test_chunk_gated_delta_rule_matches_library_recurrent(T, dtype):
    inputs = _build_model_like_inputs(T, dtype=dtype)
    chunk_out, chunk_final = _run_chunk(
        inputs, output_final_state=True, head_first=True
    )
    recurrent_out, recurrent_final = _run_recurrent(inputs)

    _assert_close(chunk_out, recurrent_out.transpose(1, 2), dtype)
    _assert_close(chunk_final, recurrent_final, dtype)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", DTYPES)
def test_chunk_gated_delta_rule_head_first_false_matches_ref(dtype):
    inputs = _build_model_like_inputs(64, dtype=dtype)
    ref_out, ref_final = _recurrent_gated_delta_rule_ref(
        inputs["q_head_first"],
        inputs["k_head_first"],
        inputs["v_head_first"],
        inputs["beta_head_first"],
        inputs["g_head_first"],
    )
    chunk_out, chunk_final = _run_chunk(
        inputs, output_final_state=True, head_first=False
    )

    _assert_close(chunk_out, ref_out.transpose(1, 2), dtype)
    _assert_close(chunk_final, ref_final, dtype)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", DTYPES)
def test_chunk_gated_delta_rule_output_final_state_flag(dtype):
    inputs = _build_model_like_inputs(64, dtype=dtype)
    ref_out, _ = _recurrent_gated_delta_rule_ref(
        inputs["q_head_first"],
        inputs["k_head_first"],
        inputs["v_head_first"],
        inputs["beta_head_first"],
        inputs["g_head_first"],
    )
    out_with_final, final_state = _run_chunk(
        inputs, output_final_state=True, head_first=True
    )
    out_without_final, final_state_none = _run_chunk(
        inputs, output_final_state=False, head_first=True
    )

    assert final_state is not None
    assert final_state_none is None
    rtol, atol = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(out_with_final, out_without_final, rtol=rtol, atol=atol)
    _assert_close(out_with_final, ref_out, dtype)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", DTYPES)
def test_chunk_gated_delta_rule_varlen(dtype):
    """cu_seqlens with two packed sequences of different lengths."""
    T1, T2 = 64, 128
    inp1 = _build_model_like_inputs(T1, dtype=dtype)
    inp2 = _build_model_like_inputs(T2, dtype=dtype)

    q = torch.cat([inp1["q_head_first"], inp2["q_head_first"]], dim=2)
    k = torch.cat([inp1["k_head_first"], inp2["k_head_first"]], dim=2)
    v = torch.cat([inp1["v_head_first"], inp2["v_head_first"]], dim=2)
    beta = torch.cat([inp1["beta_head_first"], inp2["beta_head_first"]], dim=2)
    g = torch.cat([inp1["g_head_first"], inp2["g_head_first"]], dim=2)

    device = flag_gems.device
    H, K, V = v.shape[1], q.shape[-1], v.shape[-1]
    cu_seqlens = torch.tensor([0, T1, T1 + T2], device=device, dtype=torch.long)
    initial_state = torch.zeros((2, H, K, V), device=device, dtype=dtype)
    scale = K**-0.5

    packed_out, _ = flag_gems.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        BT=64,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        head_first=True,
        scale=scale,
    )

    ref_out1, _ = _recurrent_gated_delta_rule_ref(
        inp1["q_head_first"],
        inp1["k_head_first"],
        inp1["v_head_first"],
        inp1["beta_head_first"],
        inp1["g_head_first"],
    )
    ref_out2, _ = _recurrent_gated_delta_rule_ref(
        inp2["q_head_first"],
        inp2["k_head_first"],
        inp2["v_head_first"],
        inp2["beta_head_first"],
        inp2["g_head_first"],
    )
    ref_out = torch.cat([ref_out1, ref_out2], dim=2)

    _assert_close(packed_out, ref_out, dtype)
