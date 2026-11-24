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

"""
Test script: verify whether int32 in unique.py causes incorrect results

Test strategy:
1. Compare the Triton implementation against native PyTorch results
2. Test inputs of different sizes, especially near and beyond the int32 boundary
3. Check the correctness of inverse_indices and counts
"""

import torch

from .unique import _unique2


def verify_unique_result(
    in0, sorted_flag=True, return_inverse=True, return_counts=True, verbose=True
):
    """
    Verify that _unique2 matches the native PyTorch implementation
    """
    # Triton implementation
    triton_out, triton_inverse, triton_counts = _unique2(
        in0,
        sorted=sorted_flag,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    # Native PyTorch implementation
    torch_out, torch_inverse, torch_counts = torch.unique(
        in0,
        sorted=sorted_flag,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    errors = []

    # Compare unique values
    if not torch.equal(triton_out, torch_out):
        diff_count = (triton_out != torch_out).sum().item()
        errors.append(f"unique values mismatch: {diff_count} elements differ")
        if verbose:
            mismatch_idx = torch.where(triton_out != torch_out)[0][:10]
            print(f"  First 10 mismatch positions: {mismatch_idx.tolist()}")
            for idx in mismatch_idx:
                print(
                    f"    Position {idx}: triton={triton_out[idx].item()}, torch={torch_out[idx].item()}"
                )

    # Compare inverse_indices
    if return_inverse:
        triton_inverse_flat = (
            triton_inverse.ravel() if triton_inverse is not None else None
        )
        torch_inverse_flat = torch_inverse.ravel()

        if triton_inverse_flat is not None:
            if not torch.equal(triton_inverse_flat, torch_inverse_flat):
                diff_count = (triton_inverse_flat != torch_inverse_flat).sum().item()
                errors.append(f"inverse_indices mismatch: {diff_count} elements differ")
                if verbose:
                    mismatch_idx = torch.where(
                        triton_inverse_flat != torch_inverse_flat
                    )[0][:10]
                    print(f"  First 10 mismatch positions: {mismatch_idx.tolist()}")
                    for idx in mismatch_idx:
                        triton_val = triton_inverse_flat[idx].item()
                        torch_val = torch_inverse_flat[idx].item()
                        print(
                            f"    Position {idx}: triton={triton_val}, torch={torch_val}"
                        )

                # Check for overflow symptoms (negative or unexpectedly large values)
                if triton_inverse_flat is not None:
                    negative_count = (triton_inverse_flat < 0).sum().item()
                    if negative_count > 0:
                        errors.append(
                            f"  inverse_indices contains {negative_count} negative values (possible int32 overflow)"
                        )

                    max_val = triton_inverse_flat.max().item()
                    min_val = triton_inverse_flat.min().item()
                    if max_val > 2**31 - 1 or min_val < -(2**31):
                        errors.append(
                            f"  inverse_indices out of int32 range: min={min_val}, max={max_val}"
                        )

    # Compare counts
    if return_counts:
        if triton_counts is not None and not torch.equal(triton_counts, torch_counts):
            diff_count = (triton_counts != torch_counts).sum().item()
            errors.append(f"counts mismatch: {diff_count} elements differ")
            if verbose:
                mismatch_idx = torch.where(triton_counts != torch_counts)[0][:10]
                print(f"  First 10 mismatch positions: {mismatch_idx.tolist()}")
                for idx in mismatch_idx:
                    print(
                        f"    Position {idx}: triton={triton_counts[idx].item()}, torch={torch_counts[idx].item()}"
                    )

            # Check that the sum of counts equals the number of input elements
            triton_sum = triton_counts.sum().item()
            torch_sum = torch_counts.sum().item()
            expected_sum = in0.numel()
            if triton_sum != expected_sum:
                errors.append(
                    f"  triton counts sum {triton_sum} != input element count {expected_sum}"
                )
            if torch_sum != expected_sum:
                errors.append(
                    f"  torch counts sum {torch_sum} != input element count {expected_sum}"
                )

    return errors


def test_various_sizes(device="cuda"):
    """Test inputs of different sizes"""
    print("=" * 80)
    print("Testing inputs of different sizes")
    print("=" * 80)

    # Test cases: different sizes
    test_sizes = [
        100,  # small
        1000,  # medium-small
        8192,  # simple_unique_flat boundary
        8193,  # just above the simple_unique_flat boundary
        10000,  # medium
        100000,  # large
        1000000,  # very large
        10000000,  # huge
        # Larger sizes can be tested if memory allows
        # 100000000,  # about 1/20 of the int32 upper bound
    ]

    for size in test_sizes:
        print(f"\nTest size: {size:,}")
        try:
            # Generate random data with a controlled number of unique values
            num_unique = min(
                size // 10 + 1, 100000
            )  # Control the number of unique values
            data = torch.randint(
                0, num_unique, (size,), dtype=torch.int64, device=device
            )

            errors = verify_unique_result(data, verbose=True)

            if errors:
                print(f"  ❌ Found {len(errors)} error(s):")
                for err in errors:
                    print(f"    - {err}")
            else:
                print("  ✅ Test passed")

        except Exception as e:
            print(f"  ⚠️ Test failed: {e}")


def test_edge_cases(device="cuda"):
    """Test edge cases"""
    print("\n" + "=" * 80)
    print("Testing edge cases")
    print("=" * 80)

    test_cases = [
        (
            "All identical elements",
            torch.zeros(10000, dtype=torch.int64, device=device),
        ),
        (
            "All distinct elements",
            torch.arange(10000, dtype=torch.int64, device=device),
        ),
        (
            "Sorted data",
            torch.sort(
                torch.randint(0, 1000, (10000,), dtype=torch.int64, device=device)
            )[0],
        ),
        (
            "Reverse-sorted data",
            torch.sort(
                torch.randint(0, 1000, (10000,), dtype=torch.int64, device=device),
                descending=True,
            )[0],
        ),
        (
            "Contains negative values",
            torch.randint(-5000, 5000, (10000,), dtype=torch.int64, device=device),
        ),
        (
            "Large values",
            torch.randint(2**30, 2**31, (10000,), dtype=torch.int64, device=device),
        ),
    ]

    for name, data in test_cases:
        print(f"\nTest: {name}")
        try:
            errors = verify_unique_result(data, verbose=True)
            if errors:
                print(f"  ❌ Found {len(errors)} error(s):")
                for err in errors:
                    print(f"    - {err}")
            else:
                print("  ✅ Test passed")
        except Exception as e:
            print(f"  ⚠️ Test failed: {e}")


def test_int32_overflow_specific(device="cuda"):
    """Specifically test cases that may cause int32 overflow"""
    print("\n" + "=" * 80)
    print("Specifically testing int32 overflow cases")
    print("=" * 80)

    # Test cases where cumsum may overflow
    # The maximum cumsum value equals the number of unique elements - 1
    # If the number of unique elements approaches 2^31, cumsum overflows

    print("\n1. Test cumsum overflow (requires many unique values)")
    # This requires a huge amount of memory and may not be runnable
    # but consistency can still be checked at smaller scales

    sizes_to_test = [
        (1000000, 1000000),  # 1M elements, 1M unique values
        (5000000, 1000000),  # 5M elements, 1M unique values
        (10000000, 100000),  # 10M elements, 100K unique values
    ]

    for total_size, num_unique in sizes_to_test:
        print(f"\n  Test: {total_size:,} elements, ~{num_unique:,} unique values")
        try:
            # Generate data with the specified number of unique values
            data = torch.randint(
                0, num_unique, (total_size,), dtype=torch.int64, device=device
            )

            errors = verify_unique_result(data, verbose=True)
            if errors:
                print(f"    ❌ Found {len(errors)} error(s):")
                for err in errors:
                    print(f"      - {err}")
            else:
                print("    ✅ Test passed")
        except Exception as e:
            print(f"    ⚠️ Test failed: {e}")

    print("\n2. Test index overflow (values in inverse_indices)")
    # inverse_indices stores cumsum values; it overflows if the number of unique values exceeds int32

    print("\n3. Test tile_sum overflow")
    # tile_sum stores the accumulated sum of each tile
    # It overflows if unique values per tile times the tile count exceeds int32


def test_specific_path(device="cuda"):
    """Test different code paths"""
    print("\n" + "=" * 80)
    print("Testing different code paths")
    print("=" * 80)

    # Path 1: simple_unique_flat (numel <= 8192)
    print("\nPath 1: simple_unique_flat (size <= 8192)")
    data = torch.randint(0, 100, (8000,), dtype=torch.int64, device=device)
    errors = verify_unique_result(data, return_inverse=True, return_counts=True)
    print(f"  Result: {'❌ errors' if errors else '✅ passed'}")
    for err in errors:
        print(f"    - {err}")

    # Path 2: sorted_indices_unique_flat (return_inverse=True, size > 8192)
    print("\nPath 2: sorted_indices_unique_flat (return_inverse=True, size > 8192)")
    data = torch.randint(0, 1000, (100000,), dtype=torch.int64, device=device)
    errors = verify_unique_result(data, return_inverse=True, return_counts=True)
    print(f"  Result: {'❌ errors' if errors else '✅ passed'}")
    for err in errors:
        print(f"    - {err}")

    # Path 3: sorted_quick_unique_flat (return_inverse=False, size > 8192)
    print("\nPath 3: sorted_quick_unique_flat (return_inverse=False, size > 8192)")
    data = torch.randint(0, 1000, (100000,), dtype=torch.int64, device=device)
    errors = verify_unique_result(data, return_inverse=False, return_counts=True)
    print(f"  Result: {'❌ errors' if errors else '✅ passed'}")
    for err in errors:
        print(f"    - {err}")


def detailed_debug(device="cuda"):
    """Detailed debug mode - print intermediate values"""
    print("\n" + "=" * 80)
    print("Detailed debug - inspect intermediate values")
    print("=" * 80)

    # Use a controlled test case
    size = 100000
    num_unique = 10000

    print(f"\nInput: size={size:,}, ~{num_unique:,} unique values")

    data = torch.randint(0, num_unique, (size,), dtype=torch.int64, device=device)

    # Triton results
    triton_out, triton_inverse, triton_counts = _unique2(
        data, sorted=True, return_inverse=True, return_counts=True
    )

    # PyTorch results
    torch_out, torch_inverse, torch_counts = torch.unique(
        data, sorted=True, return_inverse=True, return_counts=True
    )

    print("\nTriton results:")
    print(f"  Number of unique values: {triton_out.numel()}")
    triton_inv_min = triton_inverse.min().item()
    triton_inv_max = triton_inverse.max().item()
    print(f"  inverse_indices range: [{triton_inv_min}, {triton_inv_max}]")
    triton_cnt_min = triton_counts.min().item()
    triton_cnt_max = triton_counts.max().item()
    print(f"  counts range: [{triton_cnt_min}, {triton_cnt_max}]")
    print(f"  counts sum: {triton_counts.sum().item()}")

    print("\nPyTorch results:")
    print(f"  Number of unique values: {torch_out.numel()}")
    torch_inv_min = torch_inverse.min().item()
    torch_inv_max = torch_inverse.max().item()
    print(f"  inverse_indices range: [{torch_inv_min}, {torch_inv_max}]")
    torch_cnt_min = torch_counts.min().item()
    torch_cnt_max = torch_counts.max().item()
    print(f"  counts range: [{torch_cnt_min}, {torch_cnt_max}]")
    print(f"  counts sum: {torch_counts.sum().item()}")

    # Verify reconstruction
    print("\nVerify reconstruction:")
    triton_reconstructed = triton_out[triton_inverse.ravel()]
    torch_reconstructed = torch_out[torch_inverse.ravel()]

    triton_match = torch.equal(triton_reconstructed, data.ravel())
    torch_match = torch.equal(torch_reconstructed, data.ravel())

    print(f"  Triton reconstruction matches original data: {triton_match}")
    print(f"  PyTorch reconstruction matches original data: {torch_match}")

    if not triton_match:
        mismatch = torch.where(triton_reconstructed != data.ravel())[0]
        print(f"  Number of Triton mismatch positions: {len(mismatch)}")
        if len(mismatch) > 0:
            idx = mismatch[0].item()
            print(f"  First mismatch: position {idx}")
            print(f"    Original value: {data.ravel()[idx].item()}")
            print(f"    inverse_index: {triton_inverse.ravel()[idx].item()}")
            print(f"    Reconstructed value: {triton_reconstructed[idx].item()}")


if __name__ == "__main__":
    import sys

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA is unavailable; please run in a GPU environment")
        sys.exit(1)

    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    # Run tests
    test_specific_path(device)
    test_various_sizes(device)
    test_edge_cases(device)
    test_int32_overflow_specific(device)
    detailed_debug(device)

    print("\n" + "=" * 80)
    print("Tests completed")
    print("=" * 80)
