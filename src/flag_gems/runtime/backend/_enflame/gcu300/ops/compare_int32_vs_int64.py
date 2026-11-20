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
Comparison test: int32 version vs int64 version vs native PyTorch
Directly verify whether the int32 change causes incorrect results
"""

import sys

import torch

# Import both versions
from .unique import _unique2 as unique_int32
from .unique_int64 import _unique2_int64 as unique_int64


def compare_results(in0, return_inverse=True, return_counts=True, verbose=True):
    """
    Compare the results of three implementations:
    1. int32 version (unique.py)
    2. int64 version (unique_int64.py)
    3. Native PyTorch
    """
    results = {}

    # int32 version
    try:
        out32, inv32, cnt32 = unique_int32(
            in0, sorted=True, return_inverse=return_inverse, return_counts=return_counts
        )
        results["int32"] = {
            "out": out32,
            "inverse": inv32,
            "counts": cnt32,
            "error": None,
        }
    except Exception as e:
        results["int32"] = {
            "out": None,
            "inverse": None,
            "counts": None,
            "error": str(e),
        }

    # int64 version
    try:
        out64, inv64, cnt64 = unique_int64(
            in0, sorted=True, return_inverse=return_inverse, return_counts=return_counts
        )
        results["int64"] = {
            "out": out64,
            "inverse": inv64,
            "counts": cnt64,
            "error": None,
        }
    except Exception as e:
        results["int64"] = {
            "out": None,
            "inverse": None,
            "counts": None,
            "error": str(e),
        }

    # Native PyTorch
    try:
        out_pt, inv_pt, cnt_pt = torch.unique(
            in0, sorted=True, return_inverse=return_inverse, return_counts=return_counts
        )
        results["pytorch"] = {
            "out": out_pt,
            "inverse": inv_pt,
            "counts": cnt_pt,
            "error": None,
        }
    except Exception as e:
        results["pytorch"] = {
            "out": None,
            "inverse": None,
            "counts": None,
            "error": str(e),
        }

    return results


def analyze_differences(results, in0, verbose=True):
    """Analyze differences among the three versions"""
    issues = []

    # Check for errors
    for name, res in results.items():
        if res["error"]:
            issues.append(f"{name} version raised an error: {res['error']}")

    if any(res["error"] for res in results.values()):
        return issues

    # Compare unique outputs
    out32 = results["int32"]["out"]
    out64 = results["int64"]["out"]
    out_pt = results["pytorch"]["out"]

    print(
        f"\n  Number of unique values: int32={out32.numel()}, int64={out64.numel()}, pytorch={out_pt.numel()}"
    )

    if out32.numel() != out_pt.numel():
        issues.append(
            f"int32 unique count mismatch: {out32.numel()} vs pytorch {out_pt.numel()}"
        )
    if out64.numel() != out_pt.numel():
        issues.append(
            f"int64 unique count mismatch: {out64.numel()} vs pytorch {out_pt.numel()}"
        )

    # Compare unique values
    if not torch.equal(out32, out_pt):
        diff = (out32 != out_pt).sum().item()
        issues.append(f"int32 unique values differ from pytorch: {diff} elements")
        if verbose and diff <= 10:
            idx = torch.where(out32 != out_pt)[0]
            for i in idx[:5]:
                print(
                    f"    Position {i}: int32={out32[i].item()}, pytorch={out_pt[i].item()}"
                )

    if not torch.equal(out64, out_pt):
        diff = (out64 != out_pt).sum().item()
        issues.append(f"int64 unique values differ from pytorch: {diff} elements")

    if not torch.equal(out32, out64):
        diff = (out32 != out64).sum().item()
        issues.append(f"★ int32 and int64 unique values differ: {diff} elements ★")

    # Compare inverse_indices
    inv32 = results["int32"]["inverse"]
    inv64 = results["int64"]["inverse"]
    inv_pt = results["pytorch"]["inverse"]

    if inv32 is not None and inv_pt is not None:
        inv32_flat = inv32.ravel()
        inv64_flat = inv64.ravel() if inv64 is not None else None
        inv_pt_flat = inv_pt.ravel()

        print("  inverse_indices range:")
        print(f"    int32: [{inv32_flat.min().item()}, {inv32_flat.max().item()}]")
        if inv64_flat is not None:
            print(f"    int64: [{inv64_flat.min().item()}, {inv64_flat.max().item()}]")
        print(f"    pytorch: [{inv_pt_flat.min().item()}, {inv_pt_flat.max().item()}]")

        # Check for int32 overflow symptoms
        if inv32_flat.min().item() < 0:
            issues.append(
                f"★ int32 inverse_indices has negative values (possible overflow): min={inv32_flat.min().item()} ★"
            )

        if not torch.equal(inv32_flat, inv_pt_flat):
            diff = (inv32_flat != inv_pt_flat).sum().item()
            issues.append(
                f"int32 inverse_indices differs from pytorch: {diff} elements"
            )

            if verbose:
                # Find the first mismatch position
                mismatch_idx = torch.where(inv32_flat != inv_pt_flat)[0]
                if len(mismatch_idx) > 0:
                    idx = mismatch_idx[0].item()
                    print(f"    First mismatch position {idx}:")
                    print(f"      Original value: {in0.ravel()[idx].item()}")
                    print(f"      int32 inverse: {inv32_flat[idx].item()}")
                    print(f"      pytorch inverse: {inv_pt_flat[idx].item()}")

        if inv64_flat is not None and not torch.equal(inv64_flat, inv_pt_flat):
            diff = (inv64_flat != inv_pt_flat).sum().item()
            issues.append(
                f"int64 inverse_indices differs from pytorch: {diff} elements"
            )

        if inv64_flat is not None and not torch.equal(inv32_flat, inv64_flat):
            diff = (inv32_flat != inv64_flat).sum().item()
            issues.append(
                f"★ int32 and int64 inverse_indices differ: {diff} elements ★"
            )

            if verbose:
                mismatch_idx = torch.where(inv32_flat != inv64_flat)[0]
                if len(mismatch_idx) > 0:
                    print("    int32 vs int64 mismatch positions (first 5):")
                    for idx in mismatch_idx[:5]:
                        int32_val = inv32_flat[idx].item()
                        int64_val = inv64_flat[idx].item()
                        print(
                            f"      Position {idx}: int32={int32_val}, int64={int64_val}"
                        )

    # Compare counts
    cnt32 = results["int32"]["counts"]
    cnt64 = results["int64"]["counts"]
    cnt_pt = results["pytorch"]["counts"]

    if cnt32 is not None and cnt_pt is not None:
        print("  counts range:")
        cnt32_min = cnt32.min().item()
        cnt32_max = cnt32.max().item()
        cnt32_sum = cnt32.sum().item()
        print(f"    int32: [{cnt32_min}, {cnt32_max}], sum={cnt32_sum}")
        if cnt64 is not None:
            print(
                f"    int64: [{cnt64.min().item()}, {cnt64.max().item()}], sum={cnt64.sum().item()}"
            )
        print(
            f"    pytorch: [{cnt_pt.min().item()}, {cnt_pt.max().item()}], sum={cnt_pt.sum().item()}"
        )

        expected_sum = in0.numel()
        if cnt32.sum().item() != expected_sum:
            issues.append(
                f"★ int32 counts sum is wrong: {cnt32.sum().item()} vs expected {expected_sum} ★"
            )

        if not torch.equal(cnt32, cnt_pt):
            diff = (cnt32 != cnt_pt).sum().item()
            issues.append(f"int32 counts differ from pytorch: {diff} elements")

        if cnt64 is not None and not torch.equal(cnt32, cnt64):
            diff = (cnt32 != cnt64).sum().item()
            issues.append(f"★ int32 and int64 counts differ: {diff} elements ★")

    # Verify reconstruction
    if inv32 is not None and inv_pt is not None:
        print("\n  Verify reconstruction (out[inverse] == input):")

        inv32_flat = inv32.ravel()
        inv64_flat = inv64.ravel() if inv64 is not None else None
        inv_pt_flat = inv_pt.ravel()
        in0_flat = in0.ravel()

        # int32 reconstruction
        try:
            reconstructed32 = out32[inv32_flat]
            match32 = torch.equal(reconstructed32, in0_flat)
            print(f"    int32 reconstruction matches: {match32}")
            if not match32:
                diff = (reconstructed32 != in0_flat).sum().item()
                issues.append(
                    f"★ int32 reconstruction failed: {diff} elements differ ★"
                )
        except Exception as e:
            issues.append(f"int32 reconstruction raised an error: {e}")

        # int64 reconstruction
        if inv64_flat is not None:
            try:
                reconstructed64 = out64[inv64_flat]
                match64 = torch.equal(reconstructed64, in0_flat)
                print(f"    int64 reconstruction matches: {match64}")
                if not match64:
                    diff = (reconstructed64 != in0_flat).sum().item()
                    issues.append(
                        f"int64 reconstruction failed: {diff} elements differ"
                    )
            except Exception as e:
                issues.append(f"int64 reconstruction raised an error: {e}")

        # pytorch reconstruction
        try:
            reconstructed_pt = out_pt[inv_pt_flat]
            match_pt = torch.equal(reconstructed_pt, in0_flat)
            print(f"    pytorch reconstruction matches: {match_pt}")
        except Exception as e:
            print(f"    pytorch reconstruction raised an error: {e}")

    return issues


def run_tests(device="cuda"):
    """Run all tests"""

    print("=" * 80)
    print("int32 vs int64 comparison test")
    print("=" * 80)

    all_issues = []

    # Test cases
    test_cases = [
        # (name, size, estimated unique count)
        ("Small", 1000, 100),
        ("Boundary 8192", 8192, 1000),
        ("Just above boundary", 8193, 1000),
        ("Medium", 100000, 10000),
        ("Large - few unique", 1000000, 1000),
        ("Large - many unique", 1000000, 100000),
        ("Very large", 10000000, 100000),
    ]

    for name, size, num_unique in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Test: {name} (size={size:,}, ~{num_unique:,} unique)")
        print("=" * 60)

        try:
            # Generate test data
            data = torch.randint(
                0, num_unique, (size,), dtype=torch.int64, device=device
            )

            # Compare results
            results = compare_results(data, return_inverse=True, return_counts=True)

            # Analyze differences
            issues = analyze_differences(results, data, verbose=True)

            if issues:
                print(f"\n  ❌ Found {len(issues)} issue(s):")
                for issue in issues:
                    print(f"    - {issue}")
                all_issues.extend([(name, issue) for issue in issues])
            else:
                print("\n  ✅ Test passed")

        except Exception as e:
            error_msg = str(e)
            print(f"\n  ⚠️ Test error: {error_msg}")
            import traceback

            traceback.print_exc()
            all_issues.append((name, f"Test error: {e}"))

    # Summary
    print("\n" + "=" * 80)
    print("Test summary")
    print("=" * 80)

    if all_issues:
        print(f"\nFound {len(all_issues)} issue(s):")

        # Group by category
        int32_issues = [i for i in all_issues if "int32" in i[1].lower() or "★" in i[1]]

        if int32_issues:
            print(f"\n*** int32-related issues ({len(int32_issues)}) ***")
            for name, issue in int32_issues:
                print(f"  [{name}] {issue}")
    else:
        print(
            "\n✅ All tests passed! int32 and int64 versions produce identical results."
        )

    return all_issues


def quick_test(size=100000, num_unique=10000, device="cuda"):
    """Quick test for a single case"""
    print(f"Quick test: size={size:,}, num_unique={num_unique:,}")

    data = torch.randint(0, num_unique, (size,), dtype=torch.int64, device=device)

    results = compare_results(data, return_inverse=True, return_counts=True)
    issues = analyze_differences(results, data, verbose=True)

    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Test passed")


if __name__ == "__main__":
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA is unavailable")
        sys.exit(1)

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    if len(sys.argv) > 1:
        # Command-line argument mode
        if sys.argv[1] == "quick":
            size = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
            num_unique = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
            quick_test(size, num_unique, device)
        else:
            run_tests(device)
    else:
        run_tests(device)
