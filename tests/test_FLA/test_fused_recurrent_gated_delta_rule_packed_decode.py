import pytest
import torch

import flag_gems

try:
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as base_fused_recurrent_gated_delta_rule_packed_decode,
    )

    SGLANG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    base_fused_recurrent_gated_delta_rule_packed_decode = None
    SGLANG_AVAILABLE = False


def is_cuda_available() -> bool:
    return torch.cuda.is_available() and flag_gems.device == "cuda"


CUDA_AVAILABLE = is_cuda_available()


class FusedRecurrentGatedDeltaRulePackedDecodeTestKit:
    """Test kit for fused_recurrent_gated_delta_rule_packed_decode.

    Provides shapes and input builders for correctness testing against
    the sglang reference implementation.
    """

    base_dtype = torch.bfloat16

    @staticmethod
    def _cases():
        cases = [
            {
                "H": 8,
                "HV": 16,
                "K": 128,
                "V": 128,
                "scale": 0.08838834764831845,
                "use_qk_l2norm": True,
                "num_slots": 1024,
            },
            {
                "H": 4,
                "HV": 8,
                "K": 64,
                "V": 64,
                "scale": 0.125,
                "use_qk_l2norm": False,
                "num_slots": 256,
            },
        ]
        return cases

    @classmethod
    def get_test_params(cls):
        return cls._cases()

    @classmethod
    def build_inputs(cls, cfg, B):
        """Build input tensors for a single test case.

        Args:
            cfg: dict with H, HV, K, V, scale, use_qk_l2norm, num_slots.
            B: batch size (number of tokens).

        Returns:
            dict of input arguments for the packed decode function.
        """
        device = flag_gems.device
        dtype = cls.base_dtype

        H, HV, K, V = cfg["H"], cfg["HV"], cfg["K"], cfg["V"]

        # Packed mixed_qkv layout:
        # [Q (H*K), K (H*K), V (HV*V)]
        qk_dim = 2 * H * K
        v_dim = HV * V
        mixed_qkv_dim = qk_dim + v_dim

        mixed_qkv = torch.randn((B, mixed_qkv_dim), device=device, dtype=dtype)
        a = torch.randn((B, HV), device=device, dtype=dtype)
        b = torch.randn((B, HV), device=device, dtype=dtype)
        A_log = torch.randn((HV,), device=device, dtype=dtype)
        dt_bias = torch.randn((HV,), device=device, dtype=dtype)

        initial_state = (
            torch.randn((cfg["num_slots"], HV, K, V), device=device, dtype=dtype) * 0.1
        )
        out = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

        # All tokens use state slot 0 for simplicity.
        ssm_state_indices = torch.zeros(B, device=device, dtype=torch.long)

        return {
            "mixed_qkv": mixed_qkv,
            "a": a,
            "b": b,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "scale": float(cfg["scale"]),
            "initial_state": initial_state,
            "out": out,
            "ssm_state_indices": ssm_state_indices,
            "use_qk_l2norm_in_kernel": cfg["use_qk_l2norm"],
        }


def _reference_packed_decode_pytorch(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    """Pure PyTorch reference for packed decode.

    Used as fallback when sglang is not available.
    """
    B = mixed_qkv.shape[0]
    qkv_dim = mixed_qkv.shape[1]
    HV, V, K = initial_state.shape[-3:]
    qk_dim = qkv_dim - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K

    for n in range(B):
        state_idx = ssm_state_indices[n].item()
        if state_idx < 0:
            out[n].zero_()
            continue

        for i_hv in range(HV):
            i_h = i_hv // (HV // H)

            # Extract Q and K
            q_start = i_h * K
            q_end = q_start + K
            bq = mixed_qkv[n, q_start:q_end].float()

            k_start = H * K + i_h * K
            k_end = k_start + K
            bk = mixed_qkv[n, k_start:k_end].float()

            # Extract V
            v_start = 2 * H * K + i_hv * V
            v_end = v_start + V
            bv = mixed_qkv[n, v_start:v_end].float()

            if use_qk_l2norm_in_kernel:
                bq = bq / (torch.sqrt(torch.sum(bq * bq)) + 1e-6)
                bk = bk / (torch.sqrt(torch.sum(bk * bk)) + 1e-6)
            bq = bq * scale

            # Gating
            a_val = a[n, i_hv].float()
            b_val = b[n, i_hv].float()
            logA_val = A_log[i_hv].float()
            dt_bias_val = dt_bias[i_hv].float()

            x = a_val + dt_bias_val
            softplus_x = torch.where(x <= 20.0, torch.log(1.0 + torch.exp(x)), x)
            g_val = -torch.exp(logA_val) * softplus_x
            decay = torch.exp(g_val)
            beta_val = torch.sigmoid(b_val)

            # Load state h [V, K]
            h = initial_state[state_idx, i_hv].float().clone()

            # Delta rule update
            h = h * decay  # [V, K]
            # bv == v, bk == k, bq == q
            bv = bv - torch.sum(h * bk.unsqueeze(0), dim=1)  # [V]
            bv = bv * beta_val  # [V]
            h = h + bv.unsqueeze(1) * bk.unsqueeze(0)  # [V, K]

            # Output
            bo = torch.sum(h * bq.unsqueeze(0), dim=1)  # [V]
            out[n, 0, i_hv] = bo.to(out.dtype)

            # Store state back
            initial_state[state_idx, i_hv] = h.to(initial_state.dtype)

    return out, initial_state


@pytest.mark.skipif(
    not (SGLANG_AVAILABLE and CUDA_AVAILABLE),
    reason="requires sglang installed and CUDA device",
)
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.parametrize(
    "cfg", FusedRecurrentGatedDeltaRulePackedDecodeTestKit.get_test_params()
)
@pytest.mark.parametrize("B", [1, 4, 8, 16])
def test_fused_recurrent_gated_delta_rule_packed_decode(cfg, B):
    """Correctness test comparing FlagGems against sglang reference."""
    kit = FusedRecurrentGatedDeltaRulePackedDecodeTestKit
    inputs = kit.build_inputs(cfg, B)

    flag_initial = inputs["initial_state"].clone()
    flag_out = inputs["out"].clone()
    base_initial = inputs["initial_state"].clone()
    base_out = inputs["out"].clone()

    flag_out, flag_final = flag_gems.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=inputs["mixed_qkv"],
        a=inputs["a"],
        b=inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=inputs["scale"],
        initial_state=flag_initial,
        out=flag_out,
        ssm_state_indices=inputs["ssm_state_indices"],
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
    )

    base_out, base_final = base_fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=inputs["mixed_qkv"],
        a=inputs["a"],
        b=inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=inputs["scale"],
        initial_state=base_initial,
        out=base_out,
        ssm_state_indices=inputs["ssm_state_indices"],
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
    )

    # Output: single-step decode, tolerance relaxed for bfloat16 recurrence.
    torch.testing.assert_close(flag_out, base_out, rtol=1e-1, atol=2e-1)
    # Final state: per-element check with relative error tolerance since
    # bfloat16 errors compound across operations.
    mask = base_final.abs() > 1e-3
    if mask.any():
        rel_err = (
            flag_final[mask].float() - base_final[mask].float()
        ).abs() / base_final[mask].float().abs()
        assert rel_err.median() < 0.1, (
            f"Median relative error on final_state too large: "
            f"{rel_err.median():.4f}"
        )


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="requires CUDA device",
)
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.parametrize(
    "cfg", FusedRecurrentGatedDeltaRulePackedDecodeTestKit.get_test_params()
)
@pytest.mark.parametrize("B", [1, 4, 8])
def test_fused_recurrent_gated_delta_rule_packed_decode_accuracy(cfg, B):
    """Self-contained accuracy test using pure PyTorch reference.

    This test does not depend on sglang and serves as a fallback verification.
    """
    kit = FusedRecurrentGatedDeltaRulePackedDecodeTestKit
    inputs = kit.build_inputs(cfg, B)

    ref_initial = inputs["initial_state"].clone()
    ref_out = inputs["out"].clone()
    flag_initial = inputs["initial_state"].clone()
    flag_out = inputs["out"].clone()

    ref_out, ref_final = _reference_packed_decode_pytorch(
        mixed_qkv=inputs["mixed_qkv"],
        a=inputs["a"],
        b=inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=inputs["scale"],
        initial_state=ref_initial,
        out=ref_out,
        ssm_state_indices=inputs["ssm_state_indices"],
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
    )

    flag_out, flag_final = flag_gems.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=inputs["mixed_qkv"],
        a=inputs["a"],
        b=inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=inputs["scale"],
        initial_state=flag_initial,
        out=flag_out,
        ssm_state_indices=inputs["ssm_state_indices"],
        use_qk_l2norm_in_kernel=inputs["use_qk_l2norm_in_kernel"],
    )

    torch.testing.assert_close(flag_out, ref_out, rtol=1e-1, atol=2e-1)
    mask = ref_final.abs() > 1e-3
    if mask.any():
        rel_err = (
            flag_final[mask].float() - ref_final[mask].float()
        ).abs() / ref_final[mask].float().abs()
        assert rel_err.median() < 0.1, (
            f"Median relative error on final_state too large: "
            f"{rel_err.median():.4f}"
        )
