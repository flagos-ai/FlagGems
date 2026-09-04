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

import flag_gems

pytestmark = [
    pytest.mark.fused_recurrent_kda,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and flag_gems.device == "cuda"),
        reason="recurrent KDA tests require CUDA",
    ),
]


def _reference_decode(
    q,
    k,
    v,
    gate,
    beta,
    state,
    state_indices,
    scale,
):
    output_dtype = v.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    gate = gate.float()
    beta = beta.float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)

    out = torch.zeros_like(v)
    final_state = state.clone()
    _, N, H, _ = q.shape
    HV = v.shape[2]
    for n in range(N):
        slot = int(state_indices[n].item())
        if slot <= 0:
            continue
        for hv in range(HV):
            h = hv // (HV // H)
            current = final_state[slot, hv]
            current = current * torch.exp(gate[0, n, hv])[None, :]
            value = v[0, n, hv] - (current * k[0, n, h][None, :]).sum(-1)
            value = value * beta[0, n, hv]
            current = current + value[:, None] * k[0, n, h][None, :]
            out[0, n, hv] = (current * q[0, n, h][None, :]).sum(-1)
            final_state[slot, hv] = current
    return out.to(output_dtype), final_state


def _assert_accuracy(actual, expected, *, max_error, relative_rmse):
    actual = actual.float()
    expected = expected.float()
    error = actual - expected
    actual_max_error = error.abs().max().item()
    actual_relative_rmse = (
        error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)
    ).item()
    assert actual_max_error <= max_error
    assert actual_relative_rmse <= relative_rmse


def _build_inputs(N, H=4, D=128, V=None, padded=False):
    device = flag_gems.device
    V = D if V is None else V
    torch.manual_seed(1234 + N)
    q = torch.randn(1, N, H, D, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, N, H, V, dtype=torch.bfloat16, device=device)
    gate = -5.0 * torch.rand(1, N, H, D, dtype=torch.float32, device=device)
    beta = torch.rand(1, N, H, dtype=torch.float32, device=device)
    state = 0.01 * torch.randn(N + 3, H, V, D, dtype=torch.float32, device=device)
    state_indices = torch.arange(1, N + 1, dtype=torch.int32, device=device)
    if padded and N >= 4:
        state_indices[-2:] = 0
    return q, k, v, gate, beta, state, state_indices


@pytest.mark.parametrize("N", [1, 32, 64, 96, 128])
@torch.inference_mode()
def test_fused_recurrent_kda_decode_matches_reference(N):
    q, k, v, gate, beta, state, state_indices = _build_inputs(N)
    expected_out, expected_state = _reference_decode(
        q,
        k,
        v,
        gate,
        beta,
        state,
        state_indices,
        q.shape[-1] ** -0.5,
    )

    actual_state = state.clone()
    actual_out, final_state = flag_gems.fused_recurrent_kda_decode(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=actual_state,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    assert final_state.data_ptr() == actual_state.data_ptr()
    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)


@pytest.mark.parametrize("N", [32, 96])
@torch.inference_mode()
def test_fused_recurrent_kda_decode_fuses_safe_gate_and_beta(N):
    q, k, v, _, _, state, state_indices = _build_inputs(N)
    H, D = q.shape[2:]
    raw_gate = torch.randn_like(q)
    raw_beta = torch.randn(1, N, H, dtype=torch.bfloat16, device=q.device)
    A_log = (0.25 * torch.randn(H, dtype=torch.float32, device=q.device)).contiguous()
    dt_bias = (
        0.1 * torch.randn(H, D, dtype=torch.float32, device=q.device)
    ).contiguous()
    lower_bound = -5.0
    gate = lower_bound * torch.sigmoid(
        A_log.exp()[None, None, :, None] * (raw_gate.float() + dt_bias[None, None])
    )
    beta = raw_beta.float().sigmoid()
    expected_out, expected_state = _reference_decode(
        q,
        k,
        v,
        gate,
        beta,
        state,
        state_indices,
        D**-0.5,
    )

    actual_state = state.clone()
    actual_out, _ = flag_gems.fused_recurrent_kda_decode(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=actual_state,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )

    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)


@torch.inference_mode()
def test_fused_recurrent_kda_decode_handles_graph_padding_and_strided_out():
    q, k, v, gate, beta, state, state_indices = _build_inputs(8, padded=True)
    expected_out, expected_state = _reference_decode(
        q,
        k,
        v,
        gate,
        beta,
        state,
        state_indices,
        q.shape[-1] ** -0.5,
    )
    output_storage = torch.full(
        (*v.shape[:-1], v.shape[-1] + 7),
        torch.nan,
        dtype=v.dtype,
        device=v.device,
    )
    out = output_storage[..., : v.shape[-1]]

    actual_state = state.clone()
    actual_out, _ = flag_gems.fused_recurrent_kda_decode(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=actual_state,
        ssm_state_indices=state_indices,
        out=out,
    )

    assert actual_out.data_ptr() == out.data_ptr()
    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)
    assert torch.isnan(output_storage[..., v.shape[-1] :]).all()
    assert torch.equal(actual_state[0], state[0])


@torch.inference_mode()
def test_fused_recurrent_kda_fwd_matches_vllm_abi():
    q, k, v, gate, beta, state, state_indices = _build_inputs(32)
    cu_seqlens = torch.arange(33, dtype=torch.int32, device=q.device)
    expected_out, expected_state = _reference_decode(
        q, k, v, gate, beta, state, state_indices, q.shape[-1] ** -0.5
    )

    actual_state = state.clone()
    actual_out, final_state = flag_gems.fused_recurrent_kda_fwd(
        q,
        k,
        v,
        gate,
        beta,
        q.shape[-1] ** -0.5,
        actual_state,
        True,
        cu_seqlens,
        state_indices,
        None,
        True,
    )

    assert final_state.data_ptr() == actual_state.data_ptr()
    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)


@torch.inference_mode()
def test_fused_recurrent_kda_wrapper_accepts_empty_graph_sequences():
    active, graph_batch = 6, 8
    q, k, v, gate, beta, state, active_indices = _build_inputs(active)
    state_indices = torch.cat(
        [
            active_indices,
            torch.zeros(graph_batch - active, dtype=torch.int32, device=q.device),
        ]
    )
    cu_seqlens = torch.cat(
        [
            torch.arange(active + 1, dtype=torch.int32, device=q.device),
            torch.full(
                (graph_batch - active,), active, dtype=torch.int32, device=q.device
            ),
        ]
    )
    expected_out, expected_state = _reference_decode(
        q, k, v, gate, beta, state, active_indices, q.shape[-1] ** -0.5
    )

    actual_state = state.clone()
    actual_out, final_state = flag_gems.fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=actual_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        safe_gate=True,
        lower_bound=-5.0,
        output_final_state=False,
    )

    assert final_state.data_ptr() == actual_state.data_ptr()
    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)
    assert torch.equal(actual_state[0], state[0])
    assert torch.equal(actual_state[active + 1 :], state[active + 1 :])


@torch.inference_mode()
def test_fused_recurrent_kda_fwd_cuda_graph_replay():
    q, k, v, gate, beta, state, state_indices = _build_inputs(32, padded=True)
    cu_seqlens = torch.arange(33, dtype=torch.int32, device=q.device)
    cu_seqlens[-2:] = 30
    expected_out, expected_state = _reference_decode(
        q, k, v, gate, beta, state, state_indices, q.shape[-1] ** -0.5
    )

    warmup_state = state.clone()
    flag_gems.fused_recurrent_kda_fwd(
        q,
        k,
        v,
        gate,
        beta,
        q.shape[-1] ** -0.5,
        warmup_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    actual_state = state.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_out, final_state = flag_gems.fused_recurrent_kda_fwd(
            q,
            k,
            v,
            gate,
            beta,
            q.shape[-1] ** -0.5,
            actual_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
    actual_state.copy_(state)
    graph.replay()
    torch.cuda.synchronize()

    assert final_state.data_ptr() == actual_state.data_ptr()
    _assert_accuracy(actual_out, expected_out, max_error=3.2e-2, relative_rmse=5e-3)
    _assert_accuracy(actual_state, expected_state, max_error=1e-4, relative_rmse=1e-4)
    assert torch.count_nonzero(actual_out[0, -2:]).item() == 0
    assert torch.equal(actual_state[0], state[0])


def test_fused_recurrent_kda_decode_rejects_k_first_state_shape():
    q, k, v, gate, beta, state, state_indices = _build_inputs(2, H=2, D=64, V=96)
    k_first = torch.empty(
        state.shape[0], 2, 64, 96, dtype=state.dtype, device=state.device
    )
    with pytest.raises(ValueError, match="V-first layout"):
        flag_gems.fused_recurrent_kda_decode(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            initial_state=k_first,
            ssm_state_indices=state_indices,
        )


def test_fused_recurrent_kda_fwd_rejects_prefill_metadata():
    q, k, v, gate, beta, state, _ = _build_inputs(2)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32, device=q.device)
    state_indices = torch.ones(1, dtype=torch.int32, device=q.device)
    with pytest.raises(ValueError, match="at least one sequence per packed token"):
        flag_gems.fused_recurrent_kda_fwd(
            q,
            k,
            v,
            gate,
            beta,
            q.shape[-1] ** -0.5,
            state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
