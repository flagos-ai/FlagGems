import random

import numpy as np
import pytest
import torch

# Import your operator
from flag_gems.fused.DSA.bin_topk import (
    bucket_sort_topk,  # Replace with actual module name
)


def assert_set_similar(actual, expected, dtype, equal_nan=False):
    """Accuracy check function, optimized for topk testing"""
    print(f"Actual shape: {actual.shape}, Expected shape: {expected.shape}")
    print(f"Actual dtype: {actual.dtype}, Expected dtype: {expected.dtype}")

    # For topk indices, we mainly care about whether the selected elements are correct
    if actual.dtype == torch.int32:
        # Calculate intersection ratio
        batch_size = actual.shape[0]
        total_intersection = 0
        total_elements = 0

        for i in range(batch_size):
            actual_set = set(actual[i].cpu().numpy())
            expected_set = set(expected[i].cpu().numpy())
            intersection = actual_set & expected_set
            intersection_ratio = len(intersection) / len(expected_set)
            total_intersection += len(intersection)
            total_elements += len(expected_set)

            print(f"Batch {i}: Intersection ratio = {intersection_ratio:.4f}")

            # Require at least 95% of topk elements to match
            assert (
                intersection_ratio >= 0.95
            ), f"Batch {i}: Only {intersection_ratio:.4f} intersection, expected at least 0.95"

        overall_ratio = total_intersection / total_elements
        print(f"Overall intersection ratio: {overall_ratio:.4f}")
        return

    # For floating point numbers, use standard comparison
    torch.testing.assert_close(
        actual, expected, atol=1e-2, rtol=1e-2, equal_nan=equal_nan
    )


def init_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_reference(tensor, requires_grad=False):
    result = tensor.detach().clone()
    if requires_grad:
        result.requires_grad_()
    return result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# """Create input data for topk operator"""
def make_topk_input(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
):
    init_seed(1234)
    inputs = torch.randn(
        (batch_size, seq_len), dtype=dtype, device=device
    ).requires_grad_(False)

    # Create starts and ends (support variable length sequences)
    starts = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Simulate variable length sequences: each batch has different sequence length
    min_len = max(1, seq_len // 2)
    ends = torch.randint(
        min_len, seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )

    return inputs, starts, ends


def reference_topk_implementation(inputs, starts, ends, topk):
    """Reference implementation - using torch.topk"""
    batch_size, seq_len = inputs.shape
    ref_indices = torch.zeros(batch_size, topk, dtype=torch.int32, device=inputs.device)

    for i in range(batch_size):
        start = starts[i].item()
        end = ends[i].item()
        seq_slice = inputs[i, start:end]

        if len(seq_slice) > 0:
            # Get topk indices
            _, topk_indices = torch.topk(seq_slice, min(topk, len(seq_slice)))
            # Convert to global indices
            global_indices = topk_indices + start
            ref_indices[i, : len(global_indices)] = global_indices

    return ref_indices


def debug_topk_results(actual, expected, inputs, test_name=""):
    """Debug topk results"""
    print(f"\n=== {test_name} ===")
    batch_size = actual.shape[0]

    m = 20
    for i in range(min(16, batch_size)):  # Only check first 2 batches
        actual_set = set(actual[i].cpu().numpy())
        expected_set = set(expected[i].cpu().numpy())
        intersection = actual_set & expected_set
        print(f"Batch {i}:")
        print(f"  Actual indices: {sorted(actual_set)[:m]}...")  # Only show first 10
        print(f"  Expected indices: {sorted(expected_set)[:m]}...")
        print(
            f"  Intersection: {len(intersection)}/{len(expected_set)} = {len(intersection) / len(expected_set):.4f}"
        )

        # Check quality of actually selected values
        actual_values = inputs[i, actual[i]].cpu().numpy()
        expected_values = inputs[i, expected[i]].cpu().numpy()

        print(
            f"  Actual top values: {np.sort(actual_values)[-m:][::-1]}"
        )  # Top 5 largest values
        print(f"  Expected top values: {np.sort(expected_values)[-m:][::-1]}")


@pytest.mark.bucket_sort_topk_forward
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [256, 1024, 8192])
@pytest.mark.parametrize("topk", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_bucket_sort_topk_forward(
    batch_size: int, seq_len: int, topk: int, dtype: torch.dtype
):
    """Bucket sort topk forward propagation test"""
    if topk > seq_len:
        pytest.skip("topk cannot be larger than seq_len")

    # Create input
    inputs, starts, ends = make_topk_input(batch_size, seq_len, dtype, device)

    # Reference implementation
    ref_indices = reference_topk_implementation(
        to_reference(inputs), to_reference(starts), to_reference(ends), topk
    )

    # Your operator implementation
    your_indices = bucket_sort_topk(inputs, starts, ends, topk)

    # Debug output
    debug_topk_results(
        your_indices, ref_indices, inputs, f"forward_b{batch_size}_s{seq_len}_k{topk}"
    )

    # Accuracy comparison - using custom topk comparison logic
    assert_set_similar(your_indices, ref_indices, dtype)


@pytest.mark.bucket_sort_topk_edge_cases
@pytest.mark.parametrize(
    "config",
    [
        # Edge case tests
        {"batch_size": 1, "seq_len": 1, "topk": 1},
        {"batch_size": 1, "seq_len": 10, "topk": 10},  # topk equals sequence length
        {"batch_size": 2, "seq_len": 100, "topk": 50},
        {"batch_size": 8, "seq_len": 17, "topk": 8},  # Small sequence
    ],
)
def test_bucket_sort_topk_edge_cases(config):
    """Bucket sort topk edge case tests"""
    dtype = torch.float32

    inputs, starts, ends = make_topk_input(
        config["batch_size"], config["seq_len"], dtype, device
    )

    # Reference implementation
    ref_indices = reference_topk_implementation(
        to_reference(inputs), to_reference(starts), to_reference(ends), config["topk"]
    )

    # Your operator implementation
    your_indices = bucket_sort_topk(inputs, starts, ends, config["topk"])

    debug_topk_results(your_indices, ref_indices, inputs, "edge_case")

    assert_set_similar(your_indices, ref_indices, dtype)


@pytest.mark.bucket_sort_topk_large_scale
@pytest.mark.parametrize(
    "config",
    [
        # Large-scale tests - using your original test parameters
        {"batch_size": 64, "seq_len": 32768, "topk": 2048},
        {"batch_size": 32, "seq_len": 65536, "topk": 4096},
        {
            "batch_size": 96,
            "seq_len": 32768,
            "topk": 2048,
        },  # Your original test parameters
    ],
)
def test_bucket_sort_topk_large_scale(config):
    """Bucket sort topk large-scale tests"""
    dtype = torch.float32

    inputs, starts, ends = make_topk_input(
        config["batch_size"], config["seq_len"], dtype, device
    )

    # Reference implementation
    ref_indices = reference_topk_implementation(
        to_reference(inputs), to_reference(starts), to_reference(ends), config["topk"]
    )

    # Your operator implementation
    your_indices = bucket_sort_topk(inputs, starts, ends, config["topk"])

    debug_topk_results(your_indices, ref_indices, inputs, "large_scale")

    assert_set_similar(your_indices, ref_indices, dtype)


@pytest.mark.bucket_sort_topk_variable_length
def test_bucket_sort_topk_variable_length():
    """Test variable length sequence processing"""
    batch_size = 4
    max_seq_len = 1024
    topk = 64
    dtype = torch.float32

    # Create input, but with different sequence lengths
    inputs = torch.randn(batch_size, max_seq_len, dtype=dtype, device=device)
    starts = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Each batch uses different sequence length
    ends = torch.tensor([100, 500, 800, 1024], dtype=torch.int32, device=device)

    # Reference implementation
    ref_indices = reference_topk_implementation(
        to_reference(inputs), to_reference(starts), to_reference(ends), topk
    )

    # Your operator implementation
    your_indices = bucket_sort_topk(inputs, starts, ends, topk)

    debug_topk_results(your_indices, ref_indices, inputs, "variable_length")

    assert_set_similar(your_indices, ref_indices, dtype)


@pytest.mark.bucket_sort_topk_correctness
def test_bucket_sort_topk_correctness():
    """Correctness test - using your original test logic"""
    batch_size = 96
    seq_len = 32768
    topk = 2048

    # torch.manual_seed(1)
    inputs = torch.randn(batch_size, seq_len, dtype=torch.float32, device=device)
    starts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    ends = torch.ones(batch_size, dtype=torch.int32, device=device) * seq_len

    # Your operator
    your_indices = bucket_sort_topk(inputs, starts, ends, topk)

    # Reference implementation
    ref_indices = torch.topk(inputs, topk, dim=-1)[1]

    # Calculate intersection ratio
    total_intersection = 0
    total_elements = 0

    for i in range(batch_size):
        your_set = set(your_indices[i].cpu().numpy())
        ref_set = set(ref_indices[i].cpu().numpy())
        intersection = your_set & ref_set
        intersection_ratio = len(intersection) / len(ref_set)
        total_intersection += len(intersection)
        total_elements += len(ref_set)

        print(f"Batch {i}: Intersection ratio = {intersection_ratio:.4f}")

        # Require at least 95% of topk elements to match
        assert (
            intersection_ratio >= 0.95
        ), f"Batch {i}: Only {intersection_ratio:.4f} intersection, expected at least 0.95"

    overall_ratio = total_intersection / total_elements
    print(f"Overall intersection ratio: {overall_ratio:.4f}")


if __name__ == "__main__":
    # Can directly run this file for testing
    pytest.main([__file__, "-v"])
