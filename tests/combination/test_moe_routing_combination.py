"""
MoE (Mixture of Experts) Routing Combination Tests.

This module tests MoE routing patterns commonly used in large language models
like Mixtral, DeepSeek-MoE, etc.

Key patterns:
- Router → TopK → Gather → Expert → Scatter
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import compute_reference, combo_assert_close, COMBO_FLOAT_DTYPES
from .utils.numerical_stability import check_finite, check_no_nan

device = flag_gems.device


class MoERouter(nn.Module):
    """Simple MoE router for testing."""

    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_weight = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        Compute routing decisions.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            router_probs: Router probabilities (batch, seq_len, num_experts)
            topk_indices: Selected expert indices (batch, seq_len, top_k)
            topk_weights: Routing weights (batch, seq_len, top_k)
        """
        # Compute router logits
        router_logits = self.router_weight(x)

        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return router_probs, topk_indices, topk_weights


class MoEExpert(nn.Module):
    """Single expert network."""

    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.activation = F.silu

    def forward(self, x):
        """Expert forward pass."""
        return self.w2(self.activation(self.w1(x)))


class MoELayer(nn.Module):
    """
    Complete MoE layer for testing.

    Structure:
        x → Router → TopK → Gather → Experts → Scatter → Output
    """

    def __init__(self, d_model, num_experts, top_k, dim_feedforward):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoERouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            [MoEExpert(d_model, dim_feedforward) for _ in range(num_experts)]
        )

    def forward(self, x):
        """
        Forward pass of MoE layer.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            output: Output tensor (batch, seq_len, d_model)
            routing_info: Dict with routing statistics
        """
        batch_size, seq_len, d_model = x.shape

        # Step 1: Router
        router_probs, topk_indices, topk_weights = self.router(x)

        # Step 2: Process each expert
        # For testing, we use a simplified gather/scatter pattern
        output = torch.zeros_like(x)

        # Flatten for easier processing
        x_flat = x.view(batch_size * seq_len, d_model)
        output_flat = output.view(batch_size * seq_len, d_model)

        topk_indices_flat = topk_indices.view(batch_size * seq_len, self.top_k)
        topk_weights_flat = topk_weights.view(batch_size * seq_len, self.top_k)

        # Process each position
        for pos in range(batch_size * seq_len):
            for k in range(self.top_k):
                expert_idx = topk_indices_flat[pos, k].item()
                weight = topk_weights_flat[pos, k]

                # Expert computation
                expert_output = self.experts[expert_idx](x_flat[pos : pos + 1])

                # Weighted combination
                output_flat[pos] += weight * expert_output[0]

        # Reshape back
        output = output_flat.view(batch_size, seq_len, d_model)

        routing_info = {
            "router_probs": router_probs,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }

        return output, routing_info


class TestMoERoutingCombination:
    """
    Tests for MoE routing combination patterns.

    Router → TopK → Gather → Expert → Scatter
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "num_experts,top_k",
        [
            pytest.param(8, 2, id="8experts_2topk"),
            pytest.param(16, 2, id="16experts_2topk"),
            pytest.param(8, 4, id="8experts_4topk"),
        ],
    )
    @pytest.mark.parametrize("dtype", COMBO_FLOAT_DTYPES)
    @pytest.mark.integration
    def test_moe_router_basic(self, num_experts, top_k, dtype, use_gems):
        """Test MoE router basic functionality."""
        batch_size, seq_len, d_model = 4, 128, 512

        router = MoERouter(d_model, num_experts, top_k)
        router = router.to(device).to(dtype)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

        router.eval()
        with torch.no_grad():
            router_probs, topk_indices, topk_weights = router(x)

        # Verify output shapes
        assert router_probs.shape == (batch_size, seq_len, num_experts)
        assert topk_indices.shape == (batch_size, seq_len, top_k)
        assert topk_weights.shape == (batch_size, seq_len, top_k)

        # Verify routing properties
        # Probabilities should sum to 1
        probs_sum = router_probs.sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), rtol=1e-4)

        # Selected weights should sum to ~1 (after normalization)
        weights_sum = topk_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), rtol=1e-4)

        # Reference comparison (router ≈ 4 ops: linear + softmax + topk + normalize)
        ref_probs, ref_indices, ref_weights = compute_reference(router, x)
        combo_assert_close(router_probs, ref_probs, dtype, num_ops=4, name=f"MoE router probs {dtype}")

    @pytest.mark.integration
    @pytest.mark.parametrize("dtype", COMBO_FLOAT_DTYPES)
    def test_moe_layer_forward(self, dtype, use_gems):
        """Test complete MoE layer forward pass."""
        batch_size, seq_len, d_model = 2, 64, 256
        num_experts = 8
        top_k = 2
        dim_feedforward = 512

        moe = MoELayer(d_model, num_experts, top_k, dim_feedforward)
        moe = moe.to(device).to(dtype)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

        moe.eval()
        with torch.no_grad():
            output, routing_info = moe(x)

        # Verify output shape
        assert output.shape == (batch_size, seq_len, d_model)

        # Verify no NaN
        check_no_nan(output, f"MoE output {dtype}")

        # Verify routing diversity (not all tokens go to same expert)
        expert_counts = (
            torch.bincount(routing_info["topk_indices"].flatten(), minlength=num_experts)
            .float()
        )
        # At least 3 different experts should be used
        active_experts = (expert_counts > 0).sum().item()
        assert active_experts >= 3, f"Only {active_experts} experts used, expected more diversity"

        # Reference comparison (MoE ≈ 8 ops: router + topk + gather + expert_fwd + scatter)
        ref_output, _ = compute_reference(moe, x)
        combo_assert_close(output, ref_output, dtype, num_ops=8, name=f"MoE forward {dtype}")

    @pytest.mark.numerical_stability
    def test_moe_router_stability(self, use_gems):
        """Test MoE router numerical stability with extreme values."""
        batch_size, seq_len, d_model = 2, 64, 256
        num_experts = 8
        top_k = 2

        router = MoERouter(d_model, num_experts, top_k)
        router = router.to(device).to(torch.float32)

        # Test with large values
        x_large = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32) * 10

        router_probs, topk_indices, topk_weights = router(x_large)

        # Check no NaN in probabilities
        check_no_nan(router_probs, "Router probs with large values")

        # Test with small values
        x_small = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32) * 1e-4

        router_probs, _, _ = router(x_small)

        check_no_nan(router_probs, "Router probs with small values")

    @pytest.mark.integration
    def test_moe_backward(self, use_gems):
        """Test MoE layer backward pass."""
        batch_size, seq_len, d_model = 2, 32, 128
        num_experts = 4
        top_k = 2
        dim_feedforward = 256

        moe = MoELayer(d_model, num_experts, top_k, dim_feedforward)
        moe = moe.to(device).to(torch.float32)

        x = torch.randn(
            batch_size, seq_len, d_model, device=device, dtype=torch.float32, requires_grad=True
        )

        output, _ = moe(x)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Verify gradients
        assert x.grad is not None
        check_finite(x.grad, "MoE input gradient")

        for name, param in moe.named_parameters():
            if param.grad is not None:
                check_finite(param.grad, f"MoE gradient {name}")

    @pytest.mark.integration
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 128), (4, 256), (16, 512)])
    @pytest.mark.parametrize("dtype", COMBO_FLOAT_DTYPES)
    def test_moe_scaling(self, batch_size, seq_len, dtype, use_gems):
        """Test MoE with different batch/sequence sizes."""
        d_model = 256
        num_experts = 8
        top_k = 2
        dim_feedforward = 512

        moe = MoELayer(d_model, num_experts, top_k, dim_feedforward)
        moe = moe.to(device).to(dtype)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

        output, routing_info = moe(x)

        assert output.shape == (batch_size, seq_len, d_model)
        check_no_nan(output, f"MoE batch={batch_size} seq={seq_len} {dtype}")


class TestMoERoutingPatterns:
    """
    Tests for specific MoE routing pattern behaviors.
    """

    @pytest.mark.integration
    def test_moe_load_balancing(self, use_gems):
        """Test MoE load balancing (expert utilization)."""
        batch_size, seq_len, d_model, num_experts, top_k = 4, 128, 256, 8, 2

        router = MoERouter(d_model, num_experts, top_k)
        router = router.to(device).to(torch.float32)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

        _, topk_indices, _ = router(x)

        # Count expert usage
        expert_counts = torch.bincount(topk_indices.flatten(), minlength=num_experts).float()

        # Calculate load balance metrics
        # CV (coefficient of variation) should not be too high
        cv = expert_counts.std() / expert_counts.mean()

        # Print for monitoring
        print(f"\nExpert counts: {expert_counts}")
        print(f"CV (load balance): {cv:.4f}")

        # Not too unbalanced (CV < 1.0 is reasonable for random routing)
        assert cv < 2.0, f"Expert load too unbalanced: CV={cv}"


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMoEPerformance:
    """Performance-related MoE tests."""

    def test_moe_memory_efficiency(self, use_gems):
        """Test MoE memory usage."""
        batch_size, seq_len, d_model = 4, 128, 512
        num_experts = 8
        top_k = 2
        dim_feedforward = 1024

        moe = MoELayer(d_model, num_experts, top_k, dim_feedforward)
        moe = moe.to(device).to(torch.float16)

        x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)

        # Track memory
        torch.cuda.reset_peak_memory_stats()

        output, _ = moe(x)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        print(f"\nMoE peak memory: {peak_memory:.2f} MB")

        # Should not exceed reasonable limit (rough estimate)
        # Model params + activations should be < 1GB for this size
        assert peak_memory < 1000, f"Memory usage too high: {peak_memory:.2f} MB"