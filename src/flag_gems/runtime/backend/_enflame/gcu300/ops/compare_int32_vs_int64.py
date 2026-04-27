"""
对比测试：int32 版本 vs int64 版本 vs PyTorch 原生
直接验证 int32 修改是否导致结果偏差
"""

import sys

import torch

# 导入两个版本
from .unique import _unique2 as unique_int32
from .unique_int64 import _unique2_int64 as unique_int64


def compare_results(in0, return_inverse=True, return_counts=True, verbose=True):
    """
    比较三个实现的结果：
    1. int32 版本 (unique.py)
    2. int64 版本 (unique_int64.py)
    3. PyTorch 原生
    """
    results = {}

    # int32 版本
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

    # int64 版本
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

    # PyTorch 原生
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
    """分析三个版本之间的差异"""
    issues = []

    # 检查是否有错误
    for name, res in results.items():
        if res["error"]:
            issues.append(f"{name} 版本出错: {res['error']}")

    if any(res["error"] for res in results.values()):
        return issues

    # 比较 unique 输出
    out32 = results["int32"]["out"]
    out64 = results["int64"]["out"]
    out_pt = results["pytorch"]["out"]

    print(
        f"\n  unique 值数量: int32={out32.numel()}, int64={out64.numel()}, pytorch={out_pt.numel()}"
    )

    if out32.numel() != out_pt.numel():
        issues.append(
            f"int32 unique 数量不匹配: {out32.numel()} vs pytorch {out_pt.numel()}"
        )
    if out64.numel() != out_pt.numel():
        issues.append(
            f"int64 unique 数量不匹配: {out64.numel()} vs pytorch {out_pt.numel()}"
        )

    # 比较 unique 值
    if not torch.equal(out32, out_pt):
        diff = (out32 != out_pt).sum().item()
        issues.append(f"int32 unique 值与 pytorch 不同: {diff} 个")
        if verbose and diff <= 10:
            idx = torch.where(out32 != out_pt)[0]
            for i in idx[:5]:
                print(
                    f"    位置 {i}: int32={out32[i].item()}, pytorch={out_pt[i].item()}"
                )

    if not torch.equal(out64, out_pt):
        diff = (out64 != out_pt).sum().item()
        issues.append(f"int64 unique 值与 pytorch 不同: {diff} 个")

    if not torch.equal(out32, out64):
        diff = (out32 != out64).sum().item()
        issues.append(f"★ int32 与 int64 unique 值不同: {diff} 个 ★")

    # 比较 inverse_indices
    inv32 = results["int32"]["inverse"]
    inv64 = results["int64"]["inverse"]
    inv_pt = results["pytorch"]["inverse"]

    if inv32 is not None and inv_pt is not None:
        inv32_flat = inv32.ravel()
        inv64_flat = inv64.ravel() if inv64 is not None else None
        inv_pt_flat = inv_pt.ravel()

        print("  inverse_indices 范围:")
        print(f"    int32: [{inv32_flat.min().item()}, {inv32_flat.max().item()}]")
        if inv64_flat is not None:
            print(f"    int64: [{inv64_flat.min().item()}, {inv64_flat.max().item()}]")
        print(f"    pytorch: [{inv_pt_flat.min().item()}, {inv_pt_flat.max().item()}]")

        # 检查 int32 溢出特征
        if inv32_flat.min().item() < 0:
            issues.append(
                f"★ int32 inverse_indices 有负值（可能溢出）: min={inv32_flat.min().item()} ★"
            )

        if not torch.equal(inv32_flat, inv_pt_flat):
            diff = (inv32_flat != inv_pt_flat).sum().item()
            issues.append(f"int32 inverse_indices 与 pytorch 不同: {diff} 个")

            if verbose:
                # 找出第一个不匹配的位置
                mismatch_idx = torch.where(inv32_flat != inv_pt_flat)[0]
                if len(mismatch_idx) > 0:
                    idx = mismatch_idx[0].item()
                    print(f"    第一个不匹配位置 {idx}:")
                    print(f"      原始值: {in0.ravel()[idx].item()}")
                    print(f"      int32 inverse: {inv32_flat[idx].item()}")
                    print(f"      pytorch inverse: {inv_pt_flat[idx].item()}")

        if inv64_flat is not None and not torch.equal(inv64_flat, inv_pt_flat):
            diff = (inv64_flat != inv_pt_flat).sum().item()
            issues.append(f"int64 inverse_indices 与 pytorch 不同: {diff} 个")

        if inv64_flat is not None and not torch.equal(inv32_flat, inv64_flat):
            diff = (inv32_flat != inv64_flat).sum().item()
            issues.append(f"★ int32 与 int64 inverse_indices 不同: {diff} 个 ★")

            if verbose:
                mismatch_idx = torch.where(inv32_flat != inv64_flat)[0]
                if len(mismatch_idx) > 0:
                    print("    int32 vs int64 差异位置（前5个）:")
                    for idx in mismatch_idx[:5]:
                        int32_val = inv32_flat[idx].item()
                        int64_val = inv64_flat[idx].item()
                        print(f"      位置 {idx}: int32={int32_val}, int64={int64_val}")

    # 比较 counts
    cnt32 = results["int32"]["counts"]
    cnt64 = results["int64"]["counts"]
    cnt_pt = results["pytorch"]["counts"]

    if cnt32 is not None and cnt_pt is not None:
        print("  counts 范围:")
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
                f"★ int32 counts 总和错误: {cnt32.sum().item()} vs 期望 {expected_sum} ★"
            )

        if not torch.equal(cnt32, cnt_pt):
            diff = (cnt32 != cnt_pt).sum().item()
            issues.append(f"int32 counts 与 pytorch 不同: {diff} 个")

        if cnt64 is not None and not torch.equal(cnt32, cnt64):
            diff = (cnt32 != cnt64).sum().item()
            issues.append(f"★ int32 与 int64 counts 不同: {diff} 个 ★")

    # 验证重建
    if inv32 is not None and inv_pt is not None:
        print("\n  验证重建 (out[inverse] == input):")

        inv32_flat = inv32.ravel()
        inv64_flat = inv64.ravel() if inv64 is not None else None
        inv_pt_flat = inv_pt.ravel()
        in0_flat = in0.ravel()

        # int32 重建
        try:
            reconstructed32 = out32[inv32_flat]
            match32 = torch.equal(reconstructed32, in0_flat)
            print(f"    int32 重建匹配: {match32}")
            if not match32:
                diff = (reconstructed32 != in0_flat).sum().item()
                issues.append(f"★ int32 重建失败: {diff} 个元素不匹配 ★")
        except Exception as e:
            issues.append(f"int32 重建出错: {e}")

        # int64 重建
        if inv64_flat is not None:
            try:
                reconstructed64 = out64[inv64_flat]
                match64 = torch.equal(reconstructed64, in0_flat)
                print(f"    int64 重建匹配: {match64}")
                if not match64:
                    diff = (reconstructed64 != in0_flat).sum().item()
                    issues.append(f"int64 重建失败: {diff} 个元素不匹配")
            except Exception as e:
                issues.append(f"int64 重建出错: {e}")

        # pytorch 重建
        try:
            reconstructed_pt = out_pt[inv_pt_flat]
            match_pt = torch.equal(reconstructed_pt, in0_flat)
            print(f"    pytorch 重建匹配: {match_pt}")
        except Exception as e:
            print(f"    pytorch 重建出错: {e}")

    return issues


def run_tests(device="cuda"):
    """运行所有测试"""

    print("=" * 80)
    print("int32 vs int64 对比测试")
    print("=" * 80)

    all_issues = []

    # 测试用例
    test_cases = [
        # (名称, 大小, unique数量估计)
        ("小规模", 1000, 100),
        ("边界 8192", 8192, 1000),
        ("刚过边界", 8193, 1000),
        ("中规模", 100000, 10000),
        ("大规模-少unique", 1000000, 1000),
        ("大规模-多unique", 1000000, 100000),
        ("很大规模", 10000000, 100000),
    ]

    for name, size, num_unique in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试: {name} (size={size:,}, ~{num_unique:,} unique)")
        print("=" * 60)

        try:
            # 生成测试数据
            data = torch.randint(
                0, num_unique, (size,), dtype=torch.int64, device=device
            )

            # 比较结果
            results = compare_results(data, return_inverse=True, return_counts=True)

            # 分析差异
            issues = analyze_differences(results, data, verbose=True)

            if issues:
                print(f"\n  ❌ 发现 {len(issues)} 个问题:")
                for issue in issues:
                    print(f"    - {issue}")
                all_issues.extend([(name, issue) for issue in issues])
            else:
                print("\n  ✅ 测试通过")

        except Exception as e:
            error_msg = str(e)
            print(f"\n  ⚠️ 测试出错: {error_msg}")
            import traceback

            traceback.print_exc()
            all_issues.append((name, f"测试出错: {e}"))

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    if all_issues:
        print(f"\n发现 {len(all_issues)} 个问题:")

        # 分类统计
        int32_issues = [i for i in all_issues if "int32" in i[1].lower() or "★" in i[1]]

        if int32_issues:
            print(f"\n*** int32 相关问题 ({len(int32_issues)} 个) ***")
            for name, issue in int32_issues:
                print(f"  [{name}] {issue}")
    else:
        print("\n✅ 所有测试通过！int32 和 int64 版本结果一致。")

    return all_issues


def quick_test(size=100000, num_unique=10000, device="cuda"):
    """快速测试单个用例"""
    print(f"快速测试: size={size:,}, num_unique={num_unique:,}")

    data = torch.randint(0, num_unique, (size,), dtype=torch.int64, device=device)

    results = compare_results(data, return_inverse=True, return_counts=True)
    issues = analyze_differences(results, data, verbose=True)

    if issues:
        print("\n❌ 发现问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 测试通过")


if __name__ == "__main__":
    # 检测设备
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA 不可用")
        sys.exit(1)

    print(f"设备: {device}")
    print(f"PyTorch: {torch.__version__}")

    if len(sys.argv) > 1:
        # 命令行参数模式
        if sys.argv[1] == "quick":
            size = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
            num_unique = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
            quick_test(size, num_unique, device)
        else:
            run_tests(device)
    else:
        run_tests(device)
