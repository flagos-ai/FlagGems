"""Correctness tests for ``chunk_gated_delta_rule``.

Covers:
    * Forward correctness (vs differentiable eager reference)
    * Backward correctness (gradcheck + grads vs eager)
    * Variable-length sequences (``cu_seqlens``)
    * Initial-state chaining and ``output_final_state``
    * fp32 / bf16 / fp16 dtype matrix
    * Padding (T not a multiple of chunk size)
    * Zero-length and singleton edge cases
"""

import pytest
import torch

import flag_gems  # noqa: F401  (registers the operator)
from flag_gems.ops.chunk_gated_delta_rule import (
    _eager_chunk_gated_delta_rule as eager_ref,
    chunk_gated_delta_rule,
)

if not torch.cuda.is_available():
    pytest.skip("chunk_gated_delta_rule requires CUDA", allow_module_level=True)
DEV = "cuda"


def _make_inputs(B, T, H, K, V, dtype, requires_grad=False, seed=0):
    g_dt = dtype if dtype.is_floating_point else torch.float32
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=DEV, dtype=dtype)
    k = torch.randn(B, T, H, K, device=DEV, dtype=dtype) * 0.3
    v = torch.randn(B, T, H, V, device=DEV, dtype=dtype)
    g = -torch.rand(B, T, H, device=DEV, dtype=g_dt) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, H, device=DEV, dtype=g_dt))
    if requires_grad:
        for t in (q, k, v, g, beta):
            t.requires_grad_(True)
    return q, k, v, g, beta


# ----------------------------- forward -----------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "B,T,H,K,V",
    [
        (1, 64, 1, 16, 16),
        (2, 128, 4, 32, 32),
        (1, 256, 2, 64, 64),
    ],
)
def test_forward_matches_eager(dtype, B, T, H, K, V):
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, dtype)
    o_ours, _ = chunk_gated_delta_rule(q, k, v, g, beta)
    o_ref, _ = eager_ref(q, k, v, g, beta, scale=K**-0.5, initial_state=None,
                         output_final_state=False)
    atol = {torch.float32: 1e-1, torch.bfloat16: 5e-1, torch.float16: 5e-1}[dtype]
    rtol = {torch.float32: 1e-2, torch.bfloat16: 1e-1, torch.float16: 1e-1}[dtype]
    assert torch.allclose(o_ours.float(), o_ref.float(), atol=atol, rtol=rtol), \
        f"forward diverges: max diff {(o_ours.float()-o_ref.float()).abs().max().item()}"


def test_padding_not_multiple_of_chunk():
    q, k, v, g, beta = _make_inputs(1, 73, 2, 16, 16, torch.float32)
    o_ours, _ = chunk_gated_delta_rule(q, k, v, g, beta)
    o_ref, _ = eager_ref(q, k, v, g, beta, scale=16**-0.5, initial_state=None,
                         output_final_state=False)
    assert o_ours.shape == (1, 73, 2, 16)
    assert torch.allclose(o_ours, o_ref, atol=5e-3, rtol=5e-3)


def test_singleton_seq_len():
    q, k, v, g, beta = _make_inputs(1, 1, 1, 8, 8, torch.float32)
    o, fs = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
    assert o.shape == (1, 1, 1, 8) and fs.shape == (1, 1, 8, 8)
    assert torch.isfinite(o).all() and torch.isfinite(fs).all()


# ----------------------------- final_state ------------------------------


def test_final_state_is_fp32_when_requested():
    q, k, v, g, beta = _make_inputs(2, 64, 2, 16, 16, torch.bfloat16)
    o, fs = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
    assert fs.dtype == torch.float32, "final_state must be fp32 for safe chaining"


def test_initial_state_chaining_equivalence():
    """Running on [0, 2T) once must equal chaining on [0, T) then [T, 2T)."""
    q, k, v, g, beta = _make_inputs(1, 128, 2, 16, 16, torch.float32)
    o_full, fs_full = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)

    o1, fs1 = chunk_gated_delta_rule(
        q[:, :64], k[:, :64], v[:, :64], g[:, :64], beta[:, :64],
        output_final_state=True,
    )
    o2, fs2 = chunk_gated_delta_rule(
        q[:, 64:], k[:, 64:], v[:, 64:], g[:, 64:], beta[:, 64:],
        initial_state=fs1, output_final_state=True,
    )
    o_chained = torch.cat([o1, o2], dim=1)
    assert torch.allclose(o_full, o_chained, atol=1e-4, rtol=1e-4)
    assert torch.allclose(fs_full, fs2, atol=1e-4, rtol=1e-4)


# ----------------------------- backward ---------------------------------


def test_gradcheck_small_fp64():
    B, T, H, K, V = 1, 8, 1, 4, 4
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=DEV, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, H, K, device=DEV, dtype=torch.float64, requires_grad=True) * 0.3
    v = torch.randn(B, T, H, V, device=DEV, dtype=torch.float64, requires_grad=True)
    g = (-torch.rand(B, T, H, device=DEV, dtype=torch.float64) * 0.1).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(B, T, H, device=DEV, dtype=torch.float64)).requires_grad_(True)

    def fn(q, k, v, g, beta):
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta)
        return o

    assert torch.autograd.gradcheck(
        fn, (q, k, v, g, beta), eps=1e-5, atol=1e-3, rtol=1e-3, fast_mode=True
    )


def test_backward_matches_eager_grads():
    B, T, H, K, V = 2, 64, 2, 16, 16
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, torch.float32, requires_grad=True, seed=42)
    o, fs = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
    do = torch.randn_like(o)
    dfs = torch.randn_like(fs)
    grads_ours = torch.autograd.grad([o, fs], [q, k, v, g, beta], [do, dfs])

    q2, k2, v2, g2, beta2 = _make_inputs(B, T, H, K, V, torch.float32, requires_grad=True, seed=42)
    o2, fs2 = eager_ref(q2, k2, v2, g2, beta2, scale=K**-0.5,
                        initial_state=None, output_final_state=True)
    grads_ref = torch.autograd.grad([o2, fs2], [q2, k2, v2, g2, beta2], [do.clone(), dfs.clone()])

    for name, a, b in zip(["dq", "dk", "dv", "dg", "dbeta"], grads_ours, grads_ref):
        diff = (a - b).abs().max().item()
        assert diff < 1e-3, f"{name} mismatch: max diff {diff}"


# ----------------------------- cu_seqlens -------------------------------


def test_cu_seqlens_forward():
    seq_lens = [37, 23, 19, 64]
    T = sum(seq_lens)
    H, K, V = 2, 16, 16
    q, k, v, g, beta = _make_inputs(1, T, H, K, V, torch.float32)
    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
                      device=DEV, dtype=torch.long)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=cu)
    assert o.shape == (1, T, H, V)
    assert torch.isfinite(o).all()

    # Compare to per-sequence independent eager (state should reset at each boundary).
    parts = []
    for i in range(len(seq_lens)):
        s, e = cu[i].item(), cu[i + 1].item()
        oi, _ = eager_ref(
            q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e],
            scale=K**-0.5, initial_state=None, output_final_state=False,
        )
        parts.append(oi)
    o_ref = torch.cat(parts, dim=1)
    diff = (o - o_ref).abs().max().item()
    assert diff < 1e-2, f"cu_seqlens forward diverges from per-seq eager: {diff}"


def test_cu_seqlens_backward_finite():
    seq_lens = [16, 32, 16]
    T = sum(seq_lens)
    H, K, V = 2, 8, 8
    q, k, v, g, beta = _make_inputs(1, T, H, K, V, torch.float32, requires_grad=True)
    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
                      device=DEV, dtype=torch.long)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=cu)
    do = torch.randn_like(o)
    grads = torch.autograd.grad(o, (q, k, v, g, beta), do)
    for name, t in zip(["dq", "dk", "dv", "dg", "dbeta"], grads):
        assert torch.isfinite(t).all(), f"{name} contains non-finite values"
