"""
Triton implementation of topk_softplus_sqrt kernel and benchmark against vLLM CUDA version.

This kernel performs:
1. Softplus activation: softplus(x) = log(1 + exp(x))
2. Sqrt transformation: sqrt(softplus(x))
3. Optional: Add correction bias for expert selection
4. Top-K selection: Select top-k experts with highest scores
5. Optional: Renormalize weights
6. Apply routed scaling factor

API strictly aligned with torch.ops._moe_C.topk_softplus_sqrt
"""

import time
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _topk_softplus_sqrt_kernel(
    gating_output_ptr,  # [num_tokens, num_experts]
    topk_weights_ptr,  # [num_tokens, topk] - output
    topk_indices_ptr,  # [num_tokens, topk] - output
    token_expert_indices_ptr,  # [num_tokens, topk] - output
    correction_bias_ptr,  # [num_experts] or None
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for topk_softplus_sqrt (standard mode).
    Each program instance processes one token (one row of gating_output).
    """
    token_idx = tl.program_id(0)

    if token_idx >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_SIZE)
    mask = expert_offsets < num_experts

    # Load gating scores
    gating_ptr = gating_output_ptr + token_idx * num_experts + expert_offsets
    scores = tl.load(gating_ptr, mask=mask, other=-float("inf"))

    # Apply softplus: log(1 + exp(x)) with numerical stability
    threshold = 20.0
    scores_float = scores.to(tl.float32)
    softplus_scores = tl.where(
        scores_float > threshold, scores_float, tl.log(1.0 + tl.exp(scores_float))
    )

    # Apply sqrt
    sqrt_scores = tl.sqrt(softplus_scores)

    # Store original scores for weight computation (before bias)
    original_scores = sqrt_scores

    # Add correction bias for expert selection if provided
    if HAS_BIAS:
        bias = tl.load(correction_bias_ptr + expert_offsets, mask=mask, other=0.0)
        sqrt_scores = sqrt_scores + bias

    # Top-K selection using iterative argmax
    scores_for_selection = sqrt_scores
    weight_sum = 0.0

    for k_idx in tl.static_range(topk):
        # Find the maximum value
        max_val = tl.max(scores_for_selection, axis=0)

        # Find the smallest index where score equals max_val
        is_max = (scores_for_selection == max_val) & mask
        idx_or_large = tl.where(is_max, expert_offsets, BLOCK_SIZE)
        max_idx = tl.min(idx_or_large, axis=0)

        # Get the weight from original scores (without bias)
        selected_weight = tl.sum(
            tl.where(expert_offsets == max_idx, original_scores, 0.0)
        )

        # Store index, weight, and token_expert_indices
        tl.store(topk_indices_ptr + token_idx * topk + k_idx, max_idx)
        tl.store(topk_weights_ptr + token_idx * topk + k_idx, selected_weight)
        # token_expert_indices = k_idx * num_tokens + token_idx (same as CUDA source_rows)
        tl.store(
            token_expert_indices_ptr + token_idx * topk + k_idx,
            k_idx * num_tokens + token_idx,
        )

        weight_sum += selected_weight

        # Zero out the selected expert for next iteration
        scores_for_selection = tl.where(
            expert_offsets == max_idx, -float("inf"), scores_for_selection
        )

    # Apply renormalization and scaling
    if renormalize:
        weight_sum = tl.where(weight_sum > 0.0, weight_sum, 1.0)
        for k_idx in tl.static_range(topk):
            w = tl.load(topk_weights_ptr + token_idx * topk + k_idx)
            w = (w / weight_sum) * routed_scaling_factor
            tl.store(topk_weights_ptr + token_idx * topk + k_idx, w)
    else:
        for k_idx in tl.static_range(topk):
            w = tl.load(topk_weights_ptr + token_idx * topk + k_idx)
            w = w * routed_scaling_factor
            tl.store(topk_weights_ptr + token_idx * topk + k_idx, w)


@triton.jit
def _topk_softplus_sqrt_hash_kernel(
    gating_output_ptr,  # [num_tokens, num_experts]
    topk_weights_ptr,  # [num_tokens, topk] - output
    topk_indices_ptr,  # [num_tokens, topk] - output
    input_ids_ptr,  # [num_tokens] - token IDs for hash lookup
    tid2eid_ptr,  # [vocab_size, topk] - token ID to expert ID mapping
    num_tokens,
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    renormalize: tl.constexpr,
    routed_scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for topk_softplus_sqrt (Hash MoE mode).
    Expert indices are predetermined from lookup table.
    """
    token_idx = tl.program_id(0)

    if token_idx >= num_tokens:
        return

    expert_offsets = tl.arange(0, BLOCK_SIZE)
    mask = expert_offsets < num_experts

    # Load gating scores
    gating_ptr = gating_output_ptr + token_idx * num_experts + expert_offsets
    scores = tl.load(gating_ptr, mask=mask, other=-float("inf"))

    # Apply softplus: log(1 + exp(x)) with numerical stability
    threshold = 20.0
    scores_float = scores.to(tl.float32)
    softplus_scores = tl.where(
        scores_float > threshold, scores_float, tl.log(1.0 + tl.exp(scores_float))
    )

    # Apply sqrt
    sqrt_scores = tl.sqrt(softplus_scores)

    # Get token ID and lookup expert indices
    token_id = tl.load(input_ids_ptr + token_idx)

    weight_sum = 0.0

    for k_idx in tl.static_range(topk):
        # Get expert index from lookup table
        expert_idx = tl.load(tid2eid_ptr + token_id * topk + k_idx)

        # Get weight for this expert
        expert_mask = expert_offsets == expert_idx
        selected_weight = tl.sum(tl.where(expert_mask, sqrt_scores, 0.0))

        # Store results (note: token_expert_indices is NOT written in Hash MoE mode per CUDA impl)
        tl.store(topk_indices_ptr + token_idx * topk + k_idx, expert_idx)
        tl.store(topk_weights_ptr + token_idx * topk + k_idx, selected_weight)

        weight_sum += selected_weight

    # Apply renormalization and scaling
    if renormalize:
        weight_sum = tl.where(weight_sum > 0.0, weight_sum, 1.0)
        for k_idx in tl.static_range(topk):
            w = tl.load(topk_weights_ptr + token_idx * topk + k_idx)
            w = (w / weight_sum) * routed_scaling_factor
            tl.store(topk_weights_ptr + token_idx * topk + k_idx, w)
    else:
        for k_idx in tl.static_range(topk):
            w = tl.load(topk_weights_ptr + token_idx * topk + k_idx)
            w = w * routed_scaling_factor
            tl.store(topk_weights_ptr + token_idx * topk + k_idx, w)


def triton_topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    tid2eid: Optional[torch.Tensor] = None,
) -> None:
    """
    Triton implementation of topk_softplus_sqrt.
    API strictly aligned with torch.ops._moe_C.topk_softplus_sqrt.

    Args:
        topk_weights: Output tensor [num_tokens, topk] for expert weights (float32)
        topk_indices: Output tensor [num_tokens, topk] for expert indices (int32)
        token_expert_indices: Output tensor [num_tokens, topk] for permute indices (int32)
        gating_output: Input tensor [num_tokens, num_experts] gating scores
        renormalize: Whether to renormalize weights to sum to 1
        routed_scaling_factor: Scaling factor applied to final weights
        e_score_correction_bias: Optional bias [num_experts] for load balancing
        input_ids: Optional token IDs [num_tokens] for Hash MoE mode
        tid2eid: Optional lookup table [vocab_size, topk] for Hash MoE mode
    """
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]

    BLOCK_SIZE = triton.next_power_of_2(num_experts)
    grid = (num_tokens,)

    use_hash = tid2eid is not None
    if use_hash:
        assert input_ids is not None, "input_ids is required for Hash MoE mode"
        # Note: token_expert_indices is NOT written in Hash MoE mode per CUDA impl
        _topk_softplus_sqrt_hash_kernel[grid](
            gating_output,
            topk_weights,
            topk_indices,
            input_ids,
            tid2eid,
            num_tokens,
            num_experts,
            topk,
            renormalize,
            routed_scaling_factor,
            BLOCK_SIZE,
        )
    else:
        _topk_softplus_sqrt_kernel[grid](
            gating_output,
            topk_weights,
            topk_indices,
            token_expert_indices,
            e_score_correction_bias
            if e_score_correction_bias is not None
            else gating_output,
            num_tokens,
            num_experts,
            topk,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias is not None,
            BLOCK_SIZE,
        )


def torch_topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    tid2eid: Optional[torch.Tensor] = None,
) -> None:
    """PyTorch reference implementation of topk_softplus_sqrt."""
    num_tokens = gating_output.shape[0]
    topk = topk_weights.shape[1]

    scores = F.softplus(gating_output.float()).sqrt()
    original_scores = scores

    use_hash = tid2eid is not None
    if use_hash:
        # Hash MoE mode: indices from lookup table
        assert input_ids is not None
        topk_ids = tid2eid[input_ids]  # [num_tokens, topk]
    else:
        # Standard mode: top-k selection
        if e_score_correction_bias is not None:
            scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
        else:
            scores_for_choice = scores
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)[1]

    weights = original_scores.gather(1, topk_ids.long())

    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        weights = weights * routed_scaling_factor

    # Compute token_expert_indices: k_idx * num_tokens + token_idx
    token_idx = torch.arange(num_tokens, device=gating_output.device).unsqueeze(1)
    k_idx = torch.arange(topk, device=gating_output.device).unsqueeze(0)
    tei = k_idx * num_tokens + token_idx

    topk_weights.copy_(weights.to(torch.float32))
    topk_indices.copy_(topk_ids.to(torch.int32))
    token_expert_indices.copy_(tei.to(torch.int32))


def vllm_topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    tid2eid: Optional[torch.Tensor] = None,
) -> None:
    """vLLM CUDA implementation wrapper."""
    import vllm._custom_ops as ops

    ops.topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        e_score_correction_bias,
        input_ids,
        tid2eid,
    )


def test_correctness():
    """Test that Triton implementation matches PyTorch reference."""
    print("=" * 60)
    print("Testing Correctness (Triton vs PyTorch Reference)")
    print("=" * 60)

    test_configs = [
        # (num_tokens, num_experts, topk, renormalize, routed_scaling_factor, has_bias, use_hash)
        (1, 128, 8, True, 1.0, False, False),
        (32, 128, 8, True, 1.0, False, False),
        (128, 256, 8, True, 1.0, False, False),
        (128, 256, 8, False, 1.0, False, False),
        (128, 256, 8, True, 1.5, False, False),
        (128, 256, 8, True, 1.0, True, False),
        (33, 384, 6, True, 1.0, True, False),
        (64, 512, 16, True, 1.0, False, False),
        # Hash MoE mode tests
        (32, 128, 8, True, 1.0, False, True),
        (64, 256, 8, False, 1.5, False, True),
        (1, 128, 8, True, 1.0, False, True),
        (128, 256, 8, True, 1.0, False, True),
        (33, 384, 6, False, 1.0, False, True),
        (64, 256, 8, True, 2.0, False, True),
    ]

    # num_tokens_list = [1, 32, 128, 33, 64]
    # num_experts_list = [128, 256, 512]
    # topk_list = [8, ]
    # renormalize_list = [True, False]
    # routed_scaling_factor_list = [1.5,]
    # has_bias_list = [True, False]
    # use_hash_list = [True, False]

    # import itertools
    # test_configs = list(itertools.product(
    #     num_tokens_list,
    #     num_experts_list,
    #     topk_list,
    #     renormalize_list,
    #     routed_scaling_factor_list,
    #     has_bias_list,
    #     use_hash_list,
    # ))

    all_passed = True

    for config in test_configs:
        (
            num_tokens,
            num_experts,
            topk,
            renormalize,
            routed_scaling_factor,
            has_bias,
            use_hash,
        ) = config
        torch.manual_seed(42)

        gating_output = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device="cuda"
        )
        e_score_correction_bias = None
        input_ids = None
        tid2eid = None

        if has_bias:
            e_score_correction_bias = (
                torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1
            )

        if use_hash:
            # Create random token IDs and lookup table
            vocab_size = 1000
            input_ids = torch.randint(
                0, vocab_size, (num_tokens,), dtype=torch.int32, device="cuda"
            )
            tid2eid = torch.randint(
                0, num_experts, (vocab_size, topk), dtype=torch.int32, device="cuda"
            )

        # Allocate output tensors for reference
        ref_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
        ref_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
        ref_tei = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")

        torch_topk_softplus_sqrt(
            ref_weights,
            ref_indices,
            ref_tei,
            gating_output,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            input_ids,
            tid2eid,
        )

        # Allocate output tensors for Triton
        triton_weights = torch.empty(
            num_tokens, topk, dtype=torch.float32, device="cuda"
        )
        triton_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
        triton_tei = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")

        triton_topk_softplus_sqrt(
            triton_weights,
            triton_indices,
            triton_tei,
            gating_output,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            input_ids,
            tid2eid,
        )

        # Sort by indices for comparison (order may differ)
        sorted_ref_ids, idx_ref = ref_indices.sort(dim=-1)
        sorted_triton_ids, idx_triton = triton_indices.sort(dim=-1)

        sorted_ref_weights = ref_weights.gather(1, idx_ref)
        sorted_triton_weights = triton_weights.gather(1, idx_triton)

        indices_match = torch.allclose(
            sorted_ref_ids.float(), sorted_triton_ids.float(), atol=0, rtol=0
        )
        weights_match = torch.allclose(
            sorted_ref_weights, sorted_triton_weights, atol=1e-4, rtol=1e-3
        )
        # Note: token_expert_indices is NOT written in Hash MoE mode per CUDA impl
        tei_match = (
            True
            if use_hash
            else torch.allclose(ref_tei.float(), triton_tei.float(), atol=0, rtol=0)
        )

        status = "PASS" if (indices_match and weights_match and tei_match) else "FAIL"
        if status == "FAIL":
            all_passed = False

        mode = "hash" if use_hash else "std"
        print(
            f"Config: tokens={num_tokens}, experts={num_experts}, topk={topk}, "
            f"renorm={renormalize}, scale={routed_scaling_factor}, bias={has_bias}, mode={mode} -> {status}"
        )

        if not indices_match:
            print(f"  Indices mismatch!")
            print(f"  Ref: {sorted_ref_ids[0]}")
            print(f"  Triton: {sorted_triton_ids[0]}")

        if not weights_match:
            max_diff = (sorted_ref_weights - sorted_triton_weights).abs().max().item()
            print(f"  Weights mismatch! Max diff: {max_diff}")

        if not tei_match:
            print(f"  token_expert_indices mismatch!")
            print(f"  Ref: {ref_tei[0]}")
            print(f"  Triton: {triton_tei[0]}")

    print()
    if all_passed:
        print("All correctness tests PASSED!")
    else:
        print("Some correctness tests FAILED!")

    return all_passed


def test_against_vllm():
    """Test that Triton implementation matches vLLM CUDA implementation."""
    print("=" * 60)
    print("Testing Against vLLM CUDA Implementation")
    print("=" * 60)

    test_configs = [
        # (num_tokens, num_experts, topk, renormalize, routed_scaling_factor, has_bias, use_hash)
        # Standard mode tests
        (1, 128, 8, True, 1.0, False, False),
        (32, 128, 8, True, 1.0, False, False),
        (128, 256, 8, True, 1.0, False, False),
        (128, 256, 8, False, 1.0, False, False),
        (128, 256, 8, True, 1.5, False, False),
        (128, 256, 8, True, 1.0, True, False),
        (33, 384, 6, True, 1.0, True, False),
        # Hash MoE mode tests
        (1, 128, 8, True, 1.0, False, True),
        (32, 128, 8, True, 1.0, False, True),
        (64, 256, 8, False, 1.5, False, True),
        (128, 256, 8, True, 1.0, False, True),
        (33, 384, 6, True, 1.0, False, True),
    ]

    all_passed = True

    for config in test_configs:
        (
            num_tokens,
            num_experts,
            topk,
            renormalize,
            routed_scaling_factor,
            has_bias,
            use_hash,
        ) = config
        torch.manual_seed(42)

        gating_output = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device="cuda"
        )
        e_score_correction_bias = None
        input_ids = None
        tid2eid = None

        if has_bias:
            e_score_correction_bias = (
                torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1
            )

        if use_hash:
            vocab_size = 1000
            input_ids = torch.randint(
                0, vocab_size, (num_tokens,), dtype=torch.int32, device="cuda"
            )
            tid2eid = torch.randint(
                0, num_experts, (vocab_size, topk), dtype=torch.int32, device="cuda"
            )

        # Allocate output tensors for vLLM
        vllm_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
        vllm_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
        vllm_tei = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")

        vllm_topk_softplus_sqrt(
            vllm_weights,
            vllm_indices,
            vllm_tei,
            gating_output,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            input_ids,
            tid2eid,
        )

        # Allocate output tensors for Triton
        triton_weights = torch.empty(
            num_tokens, topk, dtype=torch.float32, device="cuda"
        )
        triton_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
        triton_tei = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")

        triton_topk_softplus_sqrt(
            triton_weights,
            triton_indices,
            triton_tei,
            gating_output,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            input_ids,
            tid2eid,
        )

        sorted_vllm_ids, idx_vllm = vllm_indices.sort(dim=-1)
        sorted_triton_ids, idx_triton = triton_indices.sort(dim=-1)

        sorted_vllm_weights = vllm_weights.gather(1, idx_vllm)
        sorted_triton_weights = triton_weights.gather(1, idx_triton)

        indices_match = torch.allclose(
            sorted_vllm_ids.float(), sorted_triton_ids.float(), atol=0, rtol=0
        )
        weights_match = torch.allclose(
            sorted_vllm_weights, sorted_triton_weights, atol=2e-2, rtol=1e-2
        )
        # Note: token_expert_indices is NOT written in Hash MoE mode per CUDA impl
        tei_match = (
            True
            if use_hash
            else torch.allclose(vllm_tei.float(), triton_tei.float(), atol=0, rtol=0)
        )

        status = "PASS" if (indices_match and weights_match and tei_match) else "FAIL"
        if status == "FAIL":
            all_passed = False

        mode_str = "hash" if use_hash else "std"
        print(
            f"Config: tokens={num_tokens}, experts={num_experts}, topk={topk}, "
            f"renorm={renormalize}, scale={routed_scaling_factor}, bias={has_bias}, mode={mode_str} -> {status}"
        )

        if not indices_match:
            print(f"  Indices mismatch!")
            print(f"  vLLM: {sorted_vllm_ids[0]}")
            print(f"  Triton: {sorted_triton_ids[0]}")

        if not weights_match:
            max_diff = (sorted_vllm_weights - sorted_triton_weights).abs().max().item()
            print(f"  Weights mismatch! Max diff: {max_diff}")

        if not tei_match:
            print(f"  token_expert_indices mismatch!")
            print(f"  vLLM: {vllm_tei[0]}")
            print(f"  Triton: {triton_tei[0]}")

    print()
    if all_passed:
        print("All vLLM comparison tests PASSED!")
    else:
        print("Some vLLM comparison tests FAILED!")

    return all_passed


def benchmark():
    """Benchmark Triton vs vLLM CUDA implementation."""
    print("=" * 60)
    print("Benchmarking Performance")
    print("=" * 60)

    # Standard mode benchmarks
    std_configs = [
        # (num_tokens, num_experts, topk, renormalize)
        (1, 256, 8, True),
        (32, 256, 8, True),
        (64, 256, 8, True),
        (128, 256, 8, True),
        (256, 256, 8, True),
        (512, 256, 8, True),
        (1024, 256, 8, True),
        (2048, 256, 8, True),
        (4096, 256, 8, True),
    ]

    # Hash MoE mode benchmarks
    hash_configs = [
        # (num_tokens, num_experts, topk, renormalize)
        (32, 256, 8, True),
        (64, 256, 8, True),
        (128, 256, 8, True),
        (256, 256, 8, True),
        (512, 256, 8, True),
        (1024, 256, 8, True),
    ]

    warmup_iters = 10
    benchmark_iters = 100

    # Standard mode benchmark
    print(f"\n[Standard Mode]")
    print(f"{'Config':<30} {'vLLM (us)':<15} {'Triton (us)':<15} {'Speedup':<10}")
    print("-" * 70)

    for num_tokens, num_experts, topk, renormalize in std_configs:
        torch.manual_seed(42)

        gating_output = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device="cuda"
        )
        routed_scaling_factor = 1.0

        # Allocate output tensors
        vllm_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
        vllm_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
        vllm_tei = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")

        triton_weights = torch.empty(
            num_tokens, topk, dtype=torch.float32, device="cuda"
        )
        triton_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
        triton_tei = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")

        # Warmup vLLM
        for _ in range(warmup_iters):
            vllm_topk_softplus_sqrt(
                vllm_weights,
                vllm_indices,
                vllm_tei,
                gating_output,
                renormalize,
                routed_scaling_factor,
                None,
                None,
                None,
            )
        torch.cuda.synchronize()

        # Benchmark vLLM
        start = time.perf_counter()
        for _ in range(benchmark_iters):
            vllm_topk_softplus_sqrt(
                vllm_weights,
                vllm_indices,
                vllm_tei,
                gating_output,
                renormalize,
                routed_scaling_factor,
                None,
                None,
                None,
            )
        torch.cuda.synchronize()
        vllm_time = (time.perf_counter() - start) / benchmark_iters * 1e6

        # Warmup Triton
        for _ in range(warmup_iters):
            triton_topk_softplus_sqrt(
                triton_weights,
                triton_indices,
                triton_tei,
                gating_output,
                renormalize,
                routed_scaling_factor,
                None,
                None,
                None,
            )
        torch.cuda.synchronize()

        # Benchmark Triton
        start = time.perf_counter()
        for _ in range(benchmark_iters):
            triton_topk_softplus_sqrt(
                triton_weights,
                triton_indices,
                triton_tei,
                gating_output,
                renormalize,
                routed_scaling_factor,
                None,
                None,
                None,
            )
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / benchmark_iters * 1e6

        speedup = vllm_time / triton_time

        config_str = f"tokens={num_tokens}, experts={num_experts}, topk={topk}"
        print(
            f"{config_str:<30} {vllm_time:<15.2f} {triton_time:<15.2f} {speedup:<10.2f}x"
        )

    # Hash MoE mode benchmark
    print(f"\n[Hash MoE Mode]")
    print(f"{'Config':<30} {'vLLM (us)':<15} {'Triton (us)':<15} {'Speedup':<10}")
    print("-" * 70)

    vocab_size = 1000

    for num_tokens, num_experts, topk, renormalize in hash_configs:
        torch.manual_seed(42)

        gating_output = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device="cuda"
        )
        routed_scaling_factor = 1.0
        input_ids = torch.randint(
            0, vocab_size, (num_tokens,), dtype=torch.int32, device="cuda"
        )
        tid2eid = torch.randint(
            0, num_experts, (vocab_size, topk), dtype=torch.int32, device="cuda"
        )

        # Allocate output tensors
        vllm_weights = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
        vllm_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
        vllm_tei = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")

        triton_weights = torch.empty(
            num_tokens, topk, dtype=torch.float32, device="cuda"
        )
        triton_indices = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")
        triton_tei = torch.empty(num_tokens, topk, dtype=torch.int32, device="cuda")

        # Warmup vLLM
        for _ in range(warmup_iters):
            vllm_topk_softplus_sqrt(
                vllm_weights,
                vllm_indices,
                vllm_tei,
                gating_output,
                renormalize,
                routed_scaling_factor,
                None,
                input_ids,
                tid2eid,
            )
        torch.cuda.synchronize()

        # Benchmark vLLM
        start = time.perf_counter()
        for _ in range(benchmark_iters):
            vllm_topk_softplus_sqrt(
                vllm_weights,
                vllm_indices,
                vllm_tei,
                gating_output,
                renormalize,
                routed_scaling_factor,
                None,
                input_ids,
                tid2eid,
            )
        torch.cuda.synchronize()
        vllm_time = (time.perf_counter() - start) / benchmark_iters * 1e6

        # Warmup Triton
        for _ in range(warmup_iters):
            triton_topk_softplus_sqrt(
                triton_weights,
                triton_indices,
                triton_tei,
                gating_output,
                renormalize,
                routed_scaling_factor,
                None,
                input_ids,
                tid2eid,
            )
        torch.cuda.synchronize()

        # Benchmark Triton
        start = time.perf_counter()
        for _ in range(benchmark_iters):
            triton_topk_softplus_sqrt(
                triton_weights,
                triton_indices,
                triton_tei,
                gating_output,
                renormalize,
                routed_scaling_factor,
                None,
                input_ids,
                tid2eid,
            )
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / benchmark_iters * 1e6

        speedup = vllm_time / triton_time

        config_str = f"tokens={num_tokens}, experts={num_experts}, topk={topk}"
        print(
            f"{config_str:<30} {vllm_time:<15.2f} {triton_time:<15.2f} {speedup:<10.2f}x"
        )


def main():
    print("Triton topk_softplus_sqrt Implementation Test & Benchmark")
    print("=" * 60)
    print()

    correctness_passed = test_correctness()
    print()

    try:
        vllm_passed = test_against_vllm()
    except Exception as e:
        print(f"vLLM comparison skipped: {e}")
        vllm_passed = False
    print()

    if correctness_passed:
        try:
            benchmark()
        except Exception as e:
            print(f"Benchmark failed: {e}")
    else:
        print("Skipping benchmark due to correctness failures")

    # benchmark()


if __name__ == "__main__":
    main()
