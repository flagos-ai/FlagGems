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

import math
import random
from typing import Dict, List

import pytest
import torch
import torch.nn.functional as F

import flag_gems

try:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule as base_fused_recurrent_gated_delta_rule,
    )

    VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    base_fused_recurrent_gated_delta_rule = None
    VLLM_AVAILABLE = False

random.seed(42)
torch.manual_seed(42)


def is_cuda_available() -> bool:
    return torch.cuda.is_available() and flag_gems.device == "cuda"


CUDA_AVAILABLE = is_cuda_available()


def rearrange_mixed_qkv(
    mixed_qkv, key_dim, value_dim, head_k_dim, head_v_dim, tp_size=1, contiguous=True
):
    query, key, value = torch.split(
        mixed_qkv,
        [
            key_dim // tp_size,
            key_dim // tp_size,
            value_dim // tp_size,
        ],
        dim=-1,
    )
    query = query.view(1, query.shape[0], -1, head_k_dim)
    key = key.view(1, key.shape[0], -1, head_k_dim)
    value = value.view(1, value.shape[0], -1, head_v_dim)
    if contiguous:
        return query.contiguous(), key.contiguous(), value.contiguous()
    else:
        return query, key, value


class FusedRecurrentGatedDeltaRuleTestKit:
    base_dtype = torch.bfloat16

    @staticmethod
    def _cases() -> List[Dict]:
        cases = [
            {  # cu_seqlens situation
                "H": 16,  # global heads(aka key_dim); local = H / tp_size = 4
                "HV": 32,  # global value heads(aka value_dim); local = HV / tp_size = 8
                "K": 128,
                "V": 128,
                "tp_size": 4,
                "beta_has_dim_v": False,
                "inplace_final_state": True,
                "use_qk_l2norm": True,
                "scale": 0.08838834764831845,
                "ssm_state_len": 4589,
                "ssm_state_indices_all_zero": True,
                "cu_seqlens_explicit": True,
            },
        ]
        return cases

    @classmethod
    def get_test_params(cls) -> List[Dict]:
        return cls._cases()

    @classmethod
    def build_inputs(cls, cfg: Dict, T, qkv_contiguous: bool) -> Dict:
        device = flag_gems.device
        dtype = cls.base_dtype
        tp_size = cfg.get("tp_size", 1)

        B = 1  # for cu_seqlens inputs, batch size is 1 and cu_seqlens is required
        cu_seqlens_len = T + 1
        key_dim = cfg["H"] * cfg["K"]  # 16 * 128 = 2048
        value_dim = cfg["HV"] * cfg["V"]  # 32 * 128 = 4096

        assert key_dim % tp_size == 0, "key_dim must be divisible by tp_size"
        assert value_dim % tp_size == 0, "value_dim must be divisible by tp_size"
        assert (key_dim // tp_size) % cfg[
            "K"
        ] == 0, "(key_dim/tp_size) must be multiple of head_k_dim"
        assert (value_dim // tp_size) % cfg[
            "V"
        ] == 0, "(value_dim/tp_size) must be multiple of head_v_dim"

        # Build mixed_qkv with explicit (T, mixed_qkv_dim) shape. For the non-contiguous
        # branch we slice a strided view from a 3D buffer to simulate a real packing.
        mixed_qkv_dim = (2 * key_dim + value_dim) // tp_size
        total_tokens = B * T  # currently B=1, so this equals T
        mixed_qkv = torch.randn(
            (total_tokens, mixed_qkv_dim), device=device, dtype=dtype
        )

        query, key, value = rearrange_mixed_qkv(
            mixed_qkv,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=cfg["K"],
            head_v_dim=cfg["V"],
            tp_size=tp_size,
            contiguous=qkv_contiguous,
        )

        HV_local = value.shape[2]

        g = F.logsigmoid(torch.randn((B, T, HV_local), device=device, dtype=dtype))
        if cfg["beta_has_dim_v"]:
            beta = torch.rand(
                B, T, HV_local, cfg["V"], device=device, dtype=dtype
            ).sigmoid()
        else:
            beta = torch.rand(B, T, HV_local, device=device, dtype=dtype).sigmoid()

        cu_seqlens = torch.arange(cu_seqlens_len, device=device, dtype=torch.long)
        initial_state = torch.zeros(
            (cfg["ssm_state_len"], HV_local, cfg["K"], cfg["V"]),
            device=device,
            dtype=dtype,
        )
        if cfg.get("ssm_state_indices_all_zero", False):
            ssm_state_indices = torch.zeros(T, device=device, dtype=torch.long)
        else:
            ssm_state_indices = torch.arange(T, device=device, dtype=torch.long)

        scale = cfg["scale"] if cfg["scale"] is not None else cfg["K"] ** -0.5

        return {
            "q": query,
            "k": key,
            "v": value,
            "g": g,
            "beta": beta,
            "scale": float(scale),
            "initial_state": initial_state,
            "cu_seqlens": cu_seqlens,
            "inplace_final_state": cfg["inplace_final_state"],
            "use_qk_l2norm_in_kernel": cfg["use_qk_l2norm"],
            "ssm_state_indices": ssm_state_indices,
        }


@pytest.mark.skipif(
    not (VLLM_AVAILABLE and CUDA_AVAILABLE),
    reason="requires vLLM installed and CUDA device",
)
@pytest.mark.fused_recurrent_gated_delta_rule
@pytest.mark.parametrize("cfg", FusedRecurrentGatedDeltaRuleTestKit.get_test_params())
@pytest.mark.parametrize("T", [1, 2, 4, 128, 512])
@pytest.mark.parametrize("qkv_contiguous", [True, False])
def test_fused_recurrent_gated_delta_rule_matches_vllm(cfg, T, qkv_contiguous):
    kit = FusedRecurrentGatedDeltaRuleTestKit
    inputs = kit.build_inputs(cfg, T, qkv_contiguous)

    flag_initial = inputs["initial_state"].clone()
    base_initial = inputs["initial_state"].clone()

    flag_out, flag_final = flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=flag_initial,
        inplace_final_state=inputs["inplace_final_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
    )

    base_out, base_final = base_fused_recurrent_gated_delta_rule(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=inputs["scale"],
        initial_state=base_initial,
        inplace_final_state=inputs["inplace_final_state"],
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
    )

    torch.testing.assert_close(flag_out, base_out, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(flag_final, base_final, rtol=1.5, atol=1.0)


def _reference_fused_recurrent_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
    inplace_final_state=True,
):
    """Pure PyTorch reference implementation for accuracy testing."""
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    N = len(cu_seqlens) - 1

    o = torch.zeros_like(v)
    if inplace_final_state:
        final_state = initial_state.clone()
    else:
        final_state = torch.zeros(T, HV, K, V, dtype=q.dtype, device=q.device)

    for n in range(N):
        bos = cu_seqlens[n].item()
        eos = cu_seqlens[n + 1].item()
        seq_len = eos - bos

        for i_hv in range(HV):
            i_h = i_hv // (HV // H)
            h = initial_state[ssm_state_indices[bos].item(), i_hv].float().clone()

            for t in range(seq_len):
                pos = bos + t
                bq = q[0, pos, i_h].float()
                bk = k[0, pos, i_h].float()
                bv = v[0, pos, i_hv].float()

                if use_qk_l2norm_in_kernel:
                    bq = bq / (bq.norm() + 1e-6)
                    bk = bk / (bk.norm() + 1e-6)
                bq = bq * scale

                bg = g[0, pos, i_hv].float()
                h = h * torch.exp(bg)

                bv = bv - (h * bk[:, None]).sum(0)
                bb = beta[0, pos, i_hv].float()
                bv = bv * bb

                h = h + bk[:, None] * bv[None, :]
                bo = (h * bq[:, None]).sum(0)
                o[0, pos, i_hv] = bo.to(o.dtype)

                state_idx = ssm_state_indices[pos].item()
                if inplace_final_state:
                    final_state[state_idx, i_hv] = h.to(final_state.dtype)
                else:
                    final_state[pos, i_hv] = h.to(final_state.dtype)

    return o, final_state


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.fused_recurrent_gated_delta_rule
@pytest.mark.parametrize("T", [1, 2, 4, 8])
@pytest.mark.parametrize("qkv_contiguous", [True, False])
@pytest.mark.parametrize("use_qk_l2norm", [True, False])
def test_fused_recurrent_gated_delta_rule_fwd_accuracy(
    T, qkv_contiguous, use_qk_l2norm
):
    """Self-contained accuracy test using a pure PyTorch reference."""
    device = flag_gems.device
    dtype = torch.bfloat16

    B = 1
    H, HV, K, V = 4, 8, 64, 64
    tp_size = 1
    key_dim = H * K
    value_dim = HV * V

    mixed_qkv_dim = (2 * key_dim + value_dim) // tp_size
    total_tokens = B * T
    mixed_qkv = torch.randn((total_tokens, mixed_qkv_dim), device=device, dtype=dtype)

    query, key, value = rearrange_mixed_qkv(
        mixed_qkv,
        key_dim=key_dim,
        value_dim=value_dim,
        head_k_dim=K,
        head_v_dim=V,
        tp_size=tp_size,
        contiguous=qkv_contiguous,
    )

    HV_local = value.shape[2]
    g = F.logsigmoid(torch.randn((B, T, HV_local), device=device, dtype=dtype))
    beta = torch.rand(B, T, HV_local, device=device, dtype=dtype).sigmoid()
    cu_seqlens = torch.arange(T + 1, device=device, dtype=torch.long)
    ssm_state_len = 128
    initial_state = (
        torch.randn((ssm_state_len, HV_local, K, V), device=device, dtype=dtype) * 0.01
    )
    ssm_state_indices = torch.zeros(T, device=device, dtype=torch.long)
    scale = K**-0.5

    ref_out, ref_final = _reference_fused_recurrent_gated_delta_rule_fwd(
        q=query.clone(),
        k=key.clone(),
        v=value.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        inplace_final_state=True,
    )

    flag_out, flag_final = flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.clone(),
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )

    torch.testing.assert_close(flag_out, ref_out, rtol=1e-1, atol=2e-1)
    # Final state accumulates over T timesteps; use per-element relative check
    # with generous tolerance since bfloat16 errors compound across recurrence.
    mask = ref_final.abs() > 1e-3
    if mask.any():
        rel_err = (
            flag_final[mask].float() - ref_final[mask].float()
        ).abs() / ref_final[mask].float().abs()
        assert (
            rel_err.median() < 0.1
        ), f"Median relative error on final_state too large: {rel_err.median():.4f}"


_INT8_H = 4
_INT8_HV = 8
_INT8_K = 128
_INT8_V = 128


def _w8a16_int8_available() -> bool:
    return flag_gems.device == "cuda"


requires_w8a16_int8_gdn = pytest.mark.skipif(
    not _w8a16_int8_available(), reason="INT8 GDN decode requires CUDA"
)


def _make_w8a16_int8_packed_inputs(num_sequences: int):
    mixed = torch.randn(
        num_sequences,
        2 * _INT8_H * _INT8_K + _INT8_HV * _INT8_V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    q, k, v = torch.split(
        mixed,
        (
            _INT8_H * _INT8_K,
            _INT8_H * _INT8_K,
            _INT8_HV * _INT8_V,
        ),
        dim=-1,
    )
    q = q.view(1, num_sequences, _INT8_H, _INT8_K)
    k = k.view(1, num_sequences, _INT8_H, _INT8_K)
    v = (0.125 * v).view(1, num_sequences, _INT8_HV, _INT8_V)
    g = torch.empty(
        1,
        num_sequences,
        _INT8_HV,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    ).uniform_(math.log(0.98), math.log(0.995))
    beta = torch.rand(
        1,
        num_sequences,
        _INT8_HV,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    cu_seqlens = torch.arange(
        num_sequences + 1, device=flag_gems.device, dtype=torch.long
    )
    state_indices = torch.arange(
        num_sequences, device=flag_gems.device, dtype=torch.long
    )
    return q, k, v, g, beta, cu_seqlens, state_indices


def _run_w8a16_int8_bf16(q, k, v, g, beta, state, cu_seqlens, state_indices):
    return flag_gems.fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=_INT8_K**-0.5,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )


def _run_w8a16_int8_reference(q, k, v, g, beta, state):
    packed = q.shape[0] == 1
    q_float = q.float()
    k_float = k.float()
    q_float *= torch.rsqrt((q_float * q_float).sum(dim=-1, keepdim=True) + 1e-6)
    k_float *= torch.rsqrt((k_float * k_float).sum(dim=-1, keepdim=True) + 1e-6)
    q_float *= _INT8_K**-0.5

    value_heads_per_key_head = _INT8_HV // _INT8_H
    q_float = q_float.repeat_interleave(value_heads_per_key_head, dim=2)
    k_float = k_float.repeat_interleave(value_heads_per_key_head, dim=2)
    if packed:
        q_float, k_float = q_float[0], k_float[0]
        v_float, g_float, beta_float = v[0], g[0], beta[0]
    else:
        q_float, k_float = q_float[:, 0], k_float[:, 0]
        v_float, g_float, beta_float = v[:, 0], g[:, 0], beta[:, 0]

    state = state.float() * torch.exp(g_float.float())[:, :, None, None]
    residual = v_float.float() - torch.einsum("nhk,nhkv->nhv", k_float, state)
    residual *= beta_float.float()[:, :, None]
    state += k_float[:, :, :, None] * residual[:, :, None, :]
    output = torch.einsum("nhk,nhkv->nhv", q_float, state)
    output = output[None] if packed else output[:, None]
    return output.to(q.dtype), state


@requires_w8a16_int8_gdn
@pytest.mark.fused_recurrent_gated_delta_rule
def test_gdn_state_int8_round_trip():
    torch.manual_seed(0)
    state = 0.125 * torch.randn(
        2,
        _INT8_HV,
        _INT8_K,
        _INT8_V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    state_int8, state_scale = flag_gems.quantize_gdn_state_int8(state.contiguous())
    actual = flag_gems.dequantize_gdn_state_int8(
        state_int8, state_scale, output_dtype=torch.float32
    )
    error = (actual - state.float()).abs()
    assert error.max().item() < 0.005
    assert error.mean().item() < 0.0015


@requires_w8a16_int8_gdn
@pytest.mark.fused_recurrent_gated_delta_rule
@pytest.mark.parametrize("num_sequences", [4, 64, 128])
def test_fused_recurrent_gated_delta_rule_w8a16_int8_accuracy(num_sequences):
    torch.manual_seed(1)
    q, k, v, g, beta, cu_seqlens, state_indices = _make_w8a16_int8_packed_inputs(
        num_sequences
    )
    initial_state = 0.02 * torch.randn(
        num_sequences,
        _INT8_HV,
        _INT8_K,
        _INT8_V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    state_int8, state_scale = flag_gems.quantize_gdn_state_int8(
        initial_state.contiguous()
    )
    state_ref = flag_gems.dequantize_gdn_state_int8(state_int8, state_scale)

    expected, state_ref = _run_w8a16_int8_reference(q, k, v, g, beta, state_ref)
    actual, state_int8, state_scale = (
        flag_gems.fused_recurrent_gated_delta_rule_w8a16_int8(
            q,
            k,
            v,
            g,
            beta,
            state_int8,
            state_scale,
            _INT8_K**-0.5,
            cu_seqlens,
            state_indices,
            True,
        )
    )
    actual_state = flag_gems.dequantize_gdn_state_int8(state_int8, state_scale)

    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-3, rtol=8e-2)
    torch.testing.assert_close(actual_state, state_ref, atol=5e-3, rtol=2.5e-1)


@requires_w8a16_int8_gdn
@pytest.mark.fused_recurrent_gated_delta_rule
def test_fused_recurrent_gated_delta_rule_w8a16_int8_repeated_updates():
    torch.manual_seed(2)
    num_sequences = 2
    state_ref = torch.zeros(
        num_sequences,
        _INT8_HV,
        _INT8_K,
        _INT8_V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    state_int8, state_scale = flag_gems.quantize_gdn_state_int8(state_ref)
    max_output_error = 0.0
    mean_output_error = 0.0

    for _ in range(64):
        q, k, v, g, beta, cu_seqlens, state_indices = _make_w8a16_int8_packed_inputs(
            num_sequences
        )
        expected, state_ref = _run_w8a16_int8_bf16(
            q, k, v, g, beta, state_ref, cu_seqlens, state_indices
        )
        actual, state_int8, state_scale = (
            flag_gems.fused_recurrent_gated_delta_rule_w8a16_int8(
                q,
                k,
                v,
                g,
                beta,
                state_int8,
                state_scale,
                _INT8_K**-0.5,
                cu_seqlens,
                state_indices,
                True,
            )
        )
        error = (actual.float() - expected.float()).abs()
        max_output_error = max(max_output_error, error.max().item())
        mean_output_error += error.mean().item()

    actual_state = flag_gems.dequantize_gdn_state_int8(state_int8, state_scale)
    state_error = (actual_state - state_ref.float()).abs()
    assert max_output_error < 0.02
    assert mean_output_error / 64 < 0.003
    assert state_error.mean().item() < 0.025


@requires_w8a16_int8_gdn
@pytest.mark.fused_recurrent_gated_delta_rule
def test_fused_recurrent_gated_delta_rule_w8a16_int8_nonpacked_accuracy():
    torch.manual_seed(3)
    q, k, v, g, beta, _, _ = _make_w8a16_int8_packed_inputs(64)
    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()
    v = v.transpose(0, 1).contiguous()
    g = g.transpose(0, 1).contiguous()
    beta = beta.transpose(0, 1).contiguous()
    initial_state = 0.02 * torch.randn(
        64,
        _INT8_HV,
        _INT8_K,
        _INT8_V,
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    state_int8, state_scale = flag_gems.quantize_gdn_state_int8(
        initial_state.contiguous()
    )
    state_ref = flag_gems.dequantize_gdn_state_int8(state_int8, state_scale)

    expected, state_ref = _run_w8a16_int8_reference(q, k, v, g, beta, state_ref)
    actual, state_int8, state_scale = (
        flag_gems.fused_recurrent_gated_delta_rule_w8a16_int8(
            q,
            k,
            v,
            g,
            beta,
            state_int8,
            state_scale,
            _INT8_K**-0.5,
        )
    )
    actual_state = flag_gems.dequantize_gdn_state_int8(state_int8, state_scale)

    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-3, rtol=8e-2)
    torch.testing.assert_close(actual_state, state_ref, atol=5e-3, rtol=2.5e-1)
