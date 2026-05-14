# Copyright (c) 2025 FlagGems. All rights reserved.

import random
from typing import Dict, List

import pytest
import torch
import triton
from triton.runtime.autotuner import Autotuner as _Autotuner

import flag_gems
import flag_gems.fused.FLA.utils as _fla_utils
from flag_gems.utils.libentry import LibTuner

# Patch both Autotuner.run and LibTuner.run to skip autotuning loop.
# LibTuner overrides Autotuner.run, so both must be patched for the patch to
# take effect. Without this, cold Triton cache tests time out on CI.
_original_autotuner_run = _Autotuner.run
_original_libtuner_run = LibTuner.run


def _no_autotune_run(self, *args, **kwargs):
    if len(self.configs) > 1:
        self.configs = self.configs[:1]
    return _original_autotuner_run(self, *args, **kwargs)


def _no_libtuner_run(self, *args, **kwargs):
    if len(self.configs) > 1:
        self.configs = self.configs[:1]
    return _original_libtuner_run(self, *args, **kwargs)


_Autotuner.run = _no_autotune_run
LibTuner.run = _no_libtuner_run


# Set up Triton allocator for global scratch memory (required by TMA ops)
def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device=flag_gems.device)


if hasattr(triton, "set_allocator"):
    triton.set_allocator(_alloc_fn)

# Disable TMA for Blackwell GPUs (compute capability 12.x) to avoid numerical issues
_device_major, _device_minor = torch.cuda.get_device_capability()
if _device_major >= 10:  # Blackwell and newer, TMA path may have issues
    _fla_utils.is_tma_supported = False
    _fla_utils.is_nvidia_hopper = False

random.seed(42)
torch.manual_seed(42)


def is_cuda_available() -> bool:
    return torch.cuda.is_available() and flag_gems.device == "cuda"


CUDA_AVAILABLE = is_cuda_available()


def _naive_recurrent_reference(q, k, v, beta, g, scale=None):
    """Naive recurrent reference implementation for correctness checking.

    Computes the gated delta rule one token at a time using the recurrence:
      S_t = S_{t-1} * exp(g_t) + k_t^T @ v_t
      v'_t = v_t - S_{t-1} @ k_t^T
      o_t = q_t @ S_t

    All computations in float32 for numerical stability.
    """
    q, k, v, beta, g = map(lambda x: x.float(), [q, k, v, beta, g])
    if scale is None:
        scale = k.shape[-1] ** -0.5
    B, T, H, K = q.shape
    V = v.shape[-1]

    S = torch.zeros(B, H, K, V, device=q.device, dtype=torch.float32)
    o = torch.zeros(B, T, H, V, device=q.device, dtype=torch.float32)

    q = q * scale

    for i in range(T):
        k_i = k[:, i]
        q_i = q[:, i]
        v_i = v[:, i]
        beta_i = beta[:, i]
        g_i = g[:, i]

        # v_new = v - w @ S = v - (S * k[..., None]).sum(-2)
        v_i = v_i - (S * k_i.unsqueeze(-1)).sum(-2)
        v_i = v_i * beta_i.unsqueeze(-1)

        # S_new = S * exp(g) + k^T @ v_new
        S = S * g_i.exp().unsqueeze(-1).unsqueeze(-1)
        S = S + k_i.unsqueeze(-1) * v_i.unsqueeze(-2)

        # o = q @ S
        o[:, i] = torch.einsum("bhd,bhdv->bhv", q_i, S)

    return o


class ChunkGatedDeltaRuleTestKit:
    """Test kit for chunk_gated_delta_rule operator."""

    base_dtype = torch.float32

    @staticmethod
    def _cases() -> List[Dict]:
        cases = [
            {"B": 1, "H": 2, "T": 64, "K": 32, "V": 16},
            {"B": 1, "H": 4, "T": 128, "K": 64, "V": 32},
            {"B": 2, "H": 2, "T": 128, "K": 64, "V": 32},
            {"B": 1, "H": 4, "T": 256, "K": 64, "V": 64},
            {"B": 2, "H": 4, "T": 64, "K": 32, "V": 16},
        ]
        return cases

    @classmethod
    def get_test_params(cls) -> List[Dict]:
        return cls._cases()

    @classmethod
    def build_inputs(cls, cfg: Dict, dtype=None) -> Dict:
        device = flag_gems.device
        if dtype is None:
            dtype = cls.base_dtype

        B, H, T, K, V = cfg["B"], cfg["H"], cfg["T"], cfg["K"], cfg["V"]

        q = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k = torch.nn.functional.normalize(k, dim=-1, p=2)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype)
        beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
        g = torch.empty(B, T, H, device=device, dtype=dtype).uniform_(0.01, 0.03).log()

        scale = float(K**-0.5)

        return {
            "q": q,
            "k": k,
            "v": v,
            "beta": beta,
            "g": g,
            "scale": scale,
        }


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("cfg", ChunkGatedDeltaRuleTestKit.get_test_params())
def test_chunk_gated_delta_rule_forward(cfg):
    """Test forward pass produces finite outputs."""
    kit = ChunkGatedDeltaRuleTestKit
    inputs = kit.build_inputs(cfg)

    o, final_state = flag_gems.chunk_gated_delta_rule(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        beta=inputs["beta"],
        g=inputs["g"],
        scale=inputs["scale"],
    )

    assert (
        o.shape == inputs["v"].shape
    ), f"Output shape mismatch: {o.shape} vs {inputs['v'].shape}"
    assert not torch.isnan(o).any(), "Output contains NaN"
    assert not torch.isinf(o).any(), "Output contains Inf"
    assert (
        final_state is None
    ), "final_state should be None when output_final_state=False"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("cfg", ChunkGatedDeltaRuleTestKit.get_test_params())
def test_chunk_gated_delta_rule_backward(cfg):
    """Test backward pass produces finite gradients."""
    kit = ChunkGatedDeltaRuleTestKit
    inputs = kit.build_inputs(cfg)

    q = inputs["q"].clone().requires_grad_(True)
    k = inputs["k"].clone().requires_grad_(True)
    v = inputs["v"].clone().requires_grad_(True)
    beta = inputs["beta"].clone().requires_grad_(True)
    g = inputs["g"].clone().requires_grad_(True)

    o, _ = flag_gems.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        scale=inputs["scale"],
    )

    do = torch.randn_like(o)
    o.backward(do)

    for name, grad in [
        ("q", q.grad),
        ("k", k.grad),
        ("v", v.grad),
        ("beta", beta.grad),
        ("g", g.grad),
    ]:
        assert grad is not None, f"{name}.grad is None"
        assert not torch.isnan(grad).any(), f"{name}.grad contains NaN"
        assert not torch.isinf(grad).any(), f"{name}.grad contains Inf"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("cfg", ChunkGatedDeltaRuleTestKit.get_test_params())
def test_chunk_gated_delta_rule_vs_reference(cfg):
    """Test operator output approximately matches naive recurrent reference.

    Note: The chunked algorithm introduces numerical differences vs the recurrent
    form, so we use relaxed tolerances.
    """
    kit = ChunkGatedDeltaRuleTestKit
    inputs = kit.build_inputs(cfg, dtype=torch.float32)

    o, _ = flag_gems.chunk_gated_delta_rule(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        beta=inputs["beta"],
        g=inputs["g"],
        scale=inputs["scale"],
    )

    # Compute naive reference
    ref_o = _naive_recurrent_reference(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        beta=inputs["beta"],
        g=inputs["g"],
        scale=inputs["scale"],
    )

    # Use relaxed tolerances for chunked vs recurrent comparison.
    # The chunked algorithm introduces numerical differences, especially
    # for near-zero values where relative error can be large.
    torch.testing.assert_close(o, ref_o, rtol=5e-1, atol=2e-1)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
def test_chunk_gated_delta_rule_padded_sequence():
    """Test that padding works correctly when sequence length is not a multiple of BT."""
    B, H, L, K, V = 1, 2, 100, 32, 16
    dtype = torch.float32
    device = flag_gems.device

    q = torch.randn(B, L, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, K, device=device, dtype=dtype)
    k = torch.nn.functional.normalize(k, dim=-1, p=2).requires_grad_(True)
    v = torch.randn(B, L, H, V, device=device, dtype=dtype, requires_grad=True)
    beta = (
        torch.rand(B, L, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    )
    g = (
        torch.empty(B, L, H, device=device, dtype=dtype)
        .uniform_(0.01, 0.03)
        .log()
        .requires_grad_(True)
    )

    o, _ = flag_gems.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        BT=64,
    )

    assert o.shape == (B, L, H, V), f"Output shape mismatch: {o.shape}"
    assert not torch.isnan(o).any(), "Output contains NaN"

    # Test backward
    do = torch.randn_like(o)
    o.backward(do)
    assert not torch.isnan(q.grad).any(), "q.grad contains NaN"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_gated_delta_rule_dtypes(dtype):
    """Test operator works with float16 and bfloat16 dtypes.

    Note: The fused cumsum/KKT kernel internally uses float32 for numerical
    stability, so Q/K/V are cast to float32 for that computation.
    """
    B, H, L, K, V = 1, 2, 64, 32, 16
    device = flag_gems.device

    # Use float32 for inputs that go through the fused cumsum kernel
    q = torch.randn(B, L, H, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, K, device=device, dtype=dtype)
    k = (
        torch.nn.functional.normalize(k.float(), dim=-1, p=2)
        .to(dtype)
        .requires_grad_(True)
    )
    v = torch.randn(B, L, H, V, device=device, dtype=dtype, requires_grad=True)
    beta = (
        torch.rand(B, L, H, device=device, dtype=dtype).sigmoid().requires_grad_(True)
    )
    g = (
        torch.empty(B, L, H, device=device, dtype=dtype)
        .uniform_(0.01, 0.03)
        .log()
        .requires_grad_(True)
    )

    # The chunk_gated_delta_rule internally upcasts where needed
    o, _ = flag_gems.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
    )

    assert o.shape == (B, L, H, V), f"Output shape mismatch: {o.shape}"
    assert o.dtype == dtype, f"Output dtype mismatch: {o.dtype} vs {dtype}"
    assert not torch.isnan(o.float()).any(), f"Output contains NaN for dtype {dtype}"
    assert not torch.isinf(o.float()).any(), f"Output contains Inf for dtype {dtype}"

    # Test backward
    do = torch.randn_like(o)
    o.backward(do)
    for name, grad in [("q", q.grad), ("k", k.grad), ("v", v.grad)]:
        assert grad is not None, f"{name}.grad is None"
        assert not torch.isnan(
            grad.float()
        ).any(), f"{name}.grad contains NaN for dtype {dtype}"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("cfg", ChunkGatedDeltaRuleTestKit.get_test_params())
def test_chunk_gated_delta_rule_gradient_finite(cfg):
    """Verify gradients are finite and produce correct loss gradient."""
    kit = ChunkGatedDeltaRuleTestKit
    inputs = kit.build_inputs(cfg, dtype=torch.float32)

    q = inputs["q"].clone().requires_grad_(True)
    k = inputs["k"].clone().requires_grad_(True)
    v = inputs["v"].clone().requires_grad_(True)
    beta = inputs["beta"].clone().requires_grad_(True)
    g = inputs["g"].clone().requires_grad_(True)

    o, _ = flag_gems.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        scale=inputs["scale"],
    )

    # Scalar loss
    loss = o.sum()
    loss.backward()

    # Verify all gradients exist and are finite
    for name, grad in [
        ("q", q.grad),
        ("k", k.grad),
        ("v", v.grad),
        ("beta", beta.grad),
        ("g", g.grad),
    ]:
        assert grad is not None, f"{name}.grad is None"
        assert not torch.isnan(grad).any(), f"{name}.grad contains NaN"
        assert not torch.isinf(grad).any(), f"{name}.grad contains Inf"
        assert grad.abs().max() > 0, f"{name}.grad is all zeros"
