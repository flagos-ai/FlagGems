"""
Precision tests for fused_marlin_moe (FlagGems Phase 2).

Phase 2 routes through the wna16 Triton kernel (fused_moe_kernel_gptq_awq)
for true fused-dequant W4A16 GEMM. Inputs are therefore real GPTQ-quantized
weights produced by vLLM's quantize_weights, not unit-scale FP16 stand-ins.

Oracle: dequantized weights run through a naive PyTorch SwiGLU MoE reference.
The wrapper sees packed uint8 weights; the reference sees the matching
fp16/bf16 w_ref returned by quantize_weights so quantization round-off is
shared by both sides.
"""
import pytest
import torch

import flag_gems
from flag_gems.fused.fused_marlin_moe import QUANT_TYPE_UINT4B8, fused_marlin_moe

# -----------------------------------------------------------------------------
# Local GPTQ uint4b8 quantization helper (self-contained, no vllm dependency).
# Matches the layout produced by vllm.quantize_weights(..., uint4b8): values
# in [-7, 7] shifted to unsigned [1, 15] and packed two-per-byte into uint8.
# -----------------------------------------------------------------------------
QUANT_TYPE_UINT4B8_TAG = "uint4b8"


def _gptq_quantize_uint4b8(w_2d, group_size):
    """
    Symmetric per-group INT4 quantization with +8 offset (GPTQ uint4b8 convention).

    Self-contained replacement for vllm.quantize_weights(w, scalar_types.uint4b8,
    group_size, False, False). Produces unpacked integer codes (each cell a
    nibble in [0, 15]) plus the exact dequantized FP reference, both compatible
    with the layout fused_moe_kernel_gptq_awq consumes.

    Args:
        w_2d: (out_dim, in_dim), fp16 or bf16.
        group_size: int, must divide in_dim.

    Returns:
        w_ref:  (out_dim, in_dim), same dtype.  Dequantized reference values.
        w_q_unsigned: (out_dim, in_dim), uint8.  Each cell a nibble in [0, 15].
        scales: (out_dim, in_dim // group_size), same dtype as w_2d.
    """
    out_dim, in_dim = w_2d.shape
    assert in_dim % group_size == 0
    ng = in_dim // group_size

    w_grouped = w_2d.reshape(out_dim, ng, group_size).to(torch.float32)
    max_abs = w_grouped.abs().amax(dim=-1, keepdim=True)
    # scale = max_abs / 7  (symmetric INT4 range [-7, 7] after +8 offset -> [1, 15])
    scales_fp = (max_abs / 7.0).clamp(min=1e-8)

    w_q_signed = torch.round(w_grouped / scales_fp).clamp(-7, 7)
    w_ref_grouped = (w_q_signed * scales_fp).to(w_2d.dtype)
    w_q_unsigned = (w_q_signed + 8).clamp(0, 15).to(torch.uint8)

    w_ref = w_ref_grouped.reshape(out_dim, in_dim)
    w_q_unsigned = w_q_unsigned.reshape(out_dim, in_dim)
    scales = scales_fp.squeeze(-1).to(w_2d.dtype)
    return w_ref, w_q_unsigned, scales


# -----------------------------------------------------------------------------
# Shape configs.
# Tuple format: (num_tokens, num_experts, hidden_size, intermediate_size, topk)
# Hard requirement: hidden_size and intermediate_size are multiples of 128
# (the wna16 group_size). Smallest legal hidden = 128.

# -----------------------------------------------------------------------------
QUICK_CONFIGS = [
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]

FULL_CONFIGS = QUICK_CONFIGS + [
    (64, 8, 256, 512, 2),
    (128, 16, 128, 256, 4),
    # Mixtral-8x7B-like
    (1, 8, 4096, 14336, 2),
    (16, 8, 4096, 14336, 2),
    (64, 8, 4096, 14336, 2),
    # DeepSeek-V3-like (TP=8 shard)
    (1, 256, 7168, 2048, 8),
    (16, 256, 7168, 2048, 8),
    (64, 256, 7168, 2048, 8),
]

GROUP_SIZE = 128


def _quantize_moe_weight(w_fp, group_size):
    """
    Apply vLLM's per-expert GPTQ quantization, returning the packed uint8
    weight and bf16/fp16 dequantized reference, in the layout fused MoE
    kernels consume.

    Args:
        w_fp: (E, out_dim, in_dim), fp16 or bf16.
    Returns:
        w_q:    (E, out_dim, in_dim // 2), uint8   (INT4 packed two-per-byte)
        w_ref:  (E, out_dim, in_dim), same dtype as w_fp  (dequantized values)
        scales: (E, out_dim, in_dim // group_size), same dtype as w_fp
    """
    E, out_dim, in_dim = w_fp.shape
    assert (
        in_dim % group_size == 0
    ), f"in_dim={in_dim} not divisible by group_size={group_size}"

    w_q = torch.empty(E, out_dim, in_dim // 2, device=w_fp.device, dtype=torch.uint8)
    w_ref = torch.empty_like(w_fp)
    scales = torch.empty(
        E,
        out_dim,
        in_dim // group_size,
        device=w_fp.device,
        dtype=w_fp.dtype,
    )
    for e in range(E):
        # Self-contained GPTQ uint4b8 quantization (no vllm dependency).
        ref_e, q_e_unpacked, sc_e = _gptq_quantize_uint4b8(w_fp[e], group_size)
        # Pack two nibbles per byte; low nibble = even, high nibble = odd.
        q_e_packed = q_e_unpacked[:, 1::2] * 16 + q_e_unpacked[:, ::2]
        w_q[e] = q_e_packed
        w_ref[e] = ref_e
        scales[e] = sc_e
    return w_q, w_ref, scales


def _make_inputs(
    num_tokens, num_experts, hidden_size, intermediate_size, topk, dtype, device
):
    """
    Build all tensors for one test case.

    Returns:
        hidden_states          (M, K)        fp16/bf16
        w1_q, w2_q             packed uint8     -> wrapper input
        w1_ref, w2_ref         fp16/bf16        -> reference GEMM input
        topk_weights, topk_ids
        w1_scale, w2_scale     3D scales matching w1_q/w2_q
    """
    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Match vLLM's magnitude (test_moe_vllm.py line 569-570): /10 keeps the
    # quantization grid well-conditioned for INT4.
    w1_fp = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        / 10.0
    )
    w2_fp = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        / 10.0
    )

    w1_q, w1_ref, w1_scale = _quantize_moe_weight(w1_fp, GROUP_SIZE)
    w2_q, w2_ref, w2_scale = _quantize_moe_weight(w2_fp, GROUP_SIZE)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    return (
        hidden_states,
        w1_q,
        w2_q,
        w1_ref,
        w2_ref,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
    )


def _reference_swiglu_moe(hidden_states, w1_ref, w2_ref, topk_weights, topk_ids):
    """Naive but obviously-correct SwiGLU MoE reference, using dequantized weights."""
    M, _ = hidden_states.shape
    _, two_N, _ = w1_ref.shape
    N = two_N // 2
    topk = topk_ids.shape[1]
    out = torch.zeros_like(hidden_states)
    for m in range(M):
        for k in range(topk):
            e = topk_ids[m, k].item()
            w_topk = topk_weights[m, k]
            x = hidden_states[m]
            gate_up = w1_ref[e] @ x
            gate, up = gate_up[:N], gate_up[N:]
            act = torch.nn.functional.silu(gate) * up
            y = w2_ref[e] @ act
            out[m] += w_topk.to(y.dtype) * y
    return out


@pytest.mark.skip(reason="Issue #3441: The operator is not stable.")
@pytest.mark.parametrize("config", QUICK_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_marlin_moe_vs_ref(config, dtype):
    """Compare fused_marlin_moe (packed INT4) against PyTorch reference (dequant)."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flag_gems.device

    (hs, w1_q, w2_q, w1_ref, w2_ref, tw, ti, w1s, w2s) = _make_inputs(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        topk,
        dtype,
        device,
    )

    result = fused_marlin_moe(
        hidden_states=hs,
        w1=w1_q,
        w2=w2_q,
        bias1=None,
        bias2=None,
        w1_scale=w1s,
        w2_scale=w2s,
        topk_weights=tw,
        topk_ids=ti,
        quant_type_id=QUANT_TYPE_UINT4B8,
    )
    ref = _reference_swiglu_moe(hs, w1_ref, w2_ref, tw, ti)
    torch.cuda.synchronize()

    rtol = 1e-1
    atol = max(5e-2, ref.abs().max().item() * 1e-3)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol)


# -----------------------------------------------------------------------------
# MVP guardrails: features the wrapper rejects must raise NotImplementedError.
# -----------------------------------------------------------------------------


def _minimal_args(device="cuda", dtype=torch.bfloat16):
    """Smallest valid arg bundle, used to probe rejection paths."""
    M, K, N, E, topk = 4, 128, 256, 4, 2
    return _make_inputs(M, E, K, N, topk, dtype, device)


def test_rejects_unsupported_quant_type():
    hs, w1_q, w2_q, _, _, tw, ti, w1s, w2s = _minimal_args()
    with pytest.raises(NotImplementedError, match="quant_type_id"):
        fused_marlin_moe(
            hidden_states=hs,
            w1=w1_q,
            w2=w2_q,
            bias1=None,
            bias2=None,
            w1_scale=w1s,
            w2_scale=w2s,
            topk_weights=tw,
            topk_ids=ti,
            quant_type_id=999,
        )


def test_rejects_act_order():
    hs, w1_q, w2_q, _, _, tw, ti, w1s, w2s = _minimal_args()
    g_idx = torch.zeros(8, dtype=torch.long, device=hs.device)
    with pytest.raises(NotImplementedError, match="act_order"):
        fused_marlin_moe(
            hidden_states=hs,
            w1=w1_q,
            w2=w2_q,
            bias1=None,
            bias2=None,
            w1_scale=w1s,
            w2_scale=w2s,
            topk_weights=tw,
            topk_ids=ti,
            quant_type_id=QUANT_TYPE_UINT4B8,
            g_idx1=g_idx,
        )


def test_rejects_fp8_input_dtype():
    hs, w1_q, w2_q, _, _, tw, ti, w1s, w2s = _minimal_args()
    with pytest.raises(NotImplementedError, match="FP8"):
        fused_marlin_moe(
            hidden_states=hs,
            w1=w1_q,
            w2=w2_q,
            bias1=None,
            bias2=None,
            w1_scale=w1s,
            w2_scale=w2s,
            topk_weights=tw,
            topk_ids=ti,
            quant_type_id=QUANT_TYPE_UINT4B8,
            input_dtype=torch.float8_e4m3fn,
        )
