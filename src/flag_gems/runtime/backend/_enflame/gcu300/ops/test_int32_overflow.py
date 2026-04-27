"""
测试脚本：验证 unique.py 中 int32 类型是否导致结果偏差

测试策略：
1. 对比 Triton 实现结果与 PyTorch 原生结果
2. 测试不同大小的输入，特别是接近和超过 int32 边界的情况
3. 检查 inverse_indices 和 counts 的正确性
"""

import torch

from .unique import _unique2


def verify_unique_result(
    in0, sorted_flag=True, return_inverse=True, return_counts=True, verbose=True
):
    """
    验证 _unique2 的结果是否与 PyTorch 原生实现一致
    """
    # Triton 实现
    triton_out, triton_inverse, triton_counts = _unique2(
        in0,
        sorted=sorted_flag,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    # PyTorch 原生实现
    torch_out, torch_inverse, torch_counts = torch.unique(
        in0,
        sorted=sorted_flag,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    errors = []

    # 比较 unique values
    if not torch.equal(triton_out, torch_out):
        diff_count = (triton_out != torch_out).sum().item()
        errors.append(f"unique values 不匹配: {diff_count} 个元素不同")
        if verbose:
            mismatch_idx = torch.where(triton_out != torch_out)[0][:10]
            print(f"  前10个不匹配位置: {mismatch_idx.tolist()}")
            for idx in mismatch_idx:
                print(
                    f"    位置 {idx}: triton={triton_out[idx].item()}, torch={torch_out[idx].item()}"
                )

    # 比较 inverse_indices
    if return_inverse:
        triton_inverse_flat = (
            triton_inverse.ravel() if triton_inverse is not None else None
        )
        torch_inverse_flat = torch_inverse.ravel()

        if triton_inverse_flat is not None:
            if not torch.equal(triton_inverse_flat, torch_inverse_flat):
                diff_count = (triton_inverse_flat != torch_inverse_flat).sum().item()
                errors.append(f"inverse_indices 不匹配: {diff_count} 个元素不同")
                if verbose:
                    mismatch_idx = torch.where(
                        triton_inverse_flat != torch_inverse_flat
                    )[0][:10]
                    print(f"  前10个不匹配位置: {mismatch_idx.tolist()}")
                    for idx in mismatch_idx:
                        triton_val = triton_inverse_flat[idx].item()
                        torch_val = torch_inverse_flat[idx].item()
                        print(f"    位置 {idx}: triton={triton_val}, torch={torch_val}")

                # 检查是否存在溢出的特征（负数或异常大的值）
                if triton_inverse_flat is not None:
                    negative_count = (triton_inverse_flat < 0).sum().item()
                    if negative_count > 0:
                        errors.append(
                            f"  inverse_indices 中有 {negative_count} 个负数值（可能是 int32 溢出）"
                        )

                    max_val = triton_inverse_flat.max().item()
                    min_val = triton_inverse_flat.min().item()
                    if max_val > 2**31 - 1 or min_val < -(2**31):
                        errors.append(
                            f"  inverse_indices 范围超出 int32: min={min_val}, max={max_val}"
                        )

    # 比较 counts
    if return_counts:
        if triton_counts is not None and not torch.equal(triton_counts, torch_counts):
            diff_count = (triton_counts != torch_counts).sum().item()
            errors.append(f"counts 不匹配: {diff_count} 个元素不同")
            if verbose:
                mismatch_idx = torch.where(triton_counts != torch_counts)[0][:10]
                print(f"  前10个不匹配位置: {mismatch_idx.tolist()}")
                for idx in mismatch_idx:
                    print(
                        f"    位置 {idx}: triton={triton_counts[idx].item()}, torch={torch_counts[idx].item()}"
                    )

            # 检查 counts 的总和是否等于输入元素数
            triton_sum = triton_counts.sum().item()
            torch_sum = torch_counts.sum().item()
            expected_sum = in0.numel()
            if triton_sum != expected_sum:
                errors.append(
                    f"  triton counts 总和 {triton_sum} != 输入元素数 {expected_sum}"
                )
            if torch_sum != expected_sum:
                errors.append(f"  torch counts 总和 {torch_sum} != 输入元素数 {expected_sum}")

    return errors


def test_various_sizes(device="cuda"):
    """测试不同大小的输入"""
    print("=" * 80)
    print("测试不同大小的输入")
    print("=" * 80)

    # 测试用例：不同大小
    test_sizes = [
        100,  # 小规模
        1000,  # 中小规模
        8192,  # simple_unique_flat 边界
        8193,  # 刚超过 simple_unique_flat 边界
        10000,  # 中规模
        100000,  # 较大规模
        1000000,  # 大规模
        10000000,  # 很大规模
        # 如果内存允许，可以测试更大的
        # 100000000,  # 接近 int32 上界的 1/20
    ]

    for size in test_sizes:
        print(f"\n测试大小: {size:,}")
        try:
            # 生成随机数据，控制 unique 数量
            num_unique = min(size // 10 + 1, 100000)  # 控制 unique 值数量
            data = torch.randint(
                0, num_unique, (size,), dtype=torch.int64, device=device
            )

            errors = verify_unique_result(data, verbose=True)

            if errors:
                print(f"  ❌ 发现 {len(errors)} 个错误:")
                for err in errors:
                    print(f"    - {err}")
            else:
                print("  ✅ 测试通过")

        except Exception as e:
            print(f"  ⚠️ 测试失败: {e}")


def test_edge_cases(device="cuda"):
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("测试边界情况")
    print("=" * 80)

    test_cases = [
        ("全相同元素", torch.zeros(10000, dtype=torch.int64, device=device)),
        ("全不同元素", torch.arange(10000, dtype=torch.int64, device=device)),
        (
            "已排序数据",
            torch.sort(
                torch.randint(0, 1000, (10000,), dtype=torch.int64, device=device)
            )[0],
        ),
        (
            "逆序数据",
            torch.sort(
                torch.randint(0, 1000, (10000,), dtype=torch.int64, device=device),
                descending=True,
            )[0],
        ),
        (
            "包含负数",
            torch.randint(-5000, 5000, (10000,), dtype=torch.int64, device=device),
        ),
        (
            "大数值",
            torch.randint(2**30, 2**31, (10000,), dtype=torch.int64, device=device),
        ),
    ]

    for name, data in test_cases:
        print(f"\n测试: {name}")
        try:
            errors = verify_unique_result(data, verbose=True)
            if errors:
                print(f"  ❌ 发现 {len(errors)} 个错误:")
                for err in errors:
                    print(f"    - {err}")
            else:
                print("  ✅ 测试通过")
        except Exception as e:
            print(f"  ⚠️ 测试失败: {e}")


def test_int32_overflow_specific(device="cuda"):
    """专门测试可能导致 int32 溢出的情况"""
    print("\n" + "=" * 80)
    print("专门测试 int32 溢出情况")
    print("=" * 80)

    # 测试 cumsum 可能溢出的情况
    # cumsum 的最大值等于 unique 元素的数量 - 1
    # 如果 unique 元素数量接近 2^31，cumsum 会溢出

    print("\n1. 测试 cumsum 溢出 (需要大量 unique 值)")
    # 这需要非常大的内存，可能无法运行
    # 但我们可以测试较小规模下的一致性

    sizes_to_test = [
        (1000000, 1000000),  # 100万元素，100万unique
        (5000000, 1000000),  # 500万元素，100万unique
        (10000000, 100000),  # 1000万元素，10万unique
    ]

    for total_size, num_unique in sizes_to_test:
        print(f"\n  测试: {total_size:,} 元素, 约 {num_unique:,} unique 值")
        try:
            # 生成数据确保有指定数量的 unique 值
            data = torch.randint(
                0, num_unique, (total_size,), dtype=torch.int64, device=device
            )

            errors = verify_unique_result(data, verbose=True)
            if errors:
                print(f"    ❌ 发现 {len(errors)} 个错误:")
                for err in errors:
                    print(f"      - {err}")
            else:
                print("    ✅ 测试通过")
        except Exception as e:
            print(f"    ⚠️ 测试失败: {e}")

    print("\n2. 测试索引溢出 (inverse_indices 中的值)")
    # inverse_indices 存储的是 cumsum 值，如果 unique 数量超过 int32 范围会溢出

    print("\n3. 测试 tile_sum 溢出")
    # tile_sum 用于存储每个 tile 的累加和
    # 如果单个 tile 中的 unique 数量乘以 tile 数量超过 int32 会溢出


def test_specific_path(device="cuda"):
    """测试不同的代码路径"""
    print("\n" + "=" * 80)
    print("测试不同的代码路径")
    print("=" * 80)

    # Path 1: simple_unique_flat (numel <= 8192)
    print("\n路径 1: simple_unique_flat (size <= 8192)")
    data = torch.randint(0, 100, (8000,), dtype=torch.int64, device=device)
    errors = verify_unique_result(data, return_inverse=True, return_counts=True)
    print(f"  结果: {'❌ 有错误' if errors else '✅ 通过'}")
    for err in errors:
        print(f"    - {err}")

    # Path 2: sorted_indices_unique_flat (return_inverse=True, size > 8192)
    print("\n路径 2: sorted_indices_unique_flat (return_inverse=True, size > 8192)")
    data = torch.randint(0, 1000, (100000,), dtype=torch.int64, device=device)
    errors = verify_unique_result(data, return_inverse=True, return_counts=True)
    print(f"  结果: {'❌ 有错误' if errors else '✅ 通过'}")
    for err in errors:
        print(f"    - {err}")

    # Path 3: sorted_quick_unique_flat (return_inverse=False, size > 8192)
    print("\n路径 3: sorted_quick_unique_flat (return_inverse=False, size > 8192)")
    data = torch.randint(0, 1000, (100000,), dtype=torch.int64, device=device)
    errors = verify_unique_result(data, return_inverse=False, return_counts=True)
    print(f"  结果: {'❌ 有错误' if errors else '✅ 通过'}")
    for err in errors:
        print(f"    - {err}")


def detailed_debug(device="cuda"):
    """详细调试模式 - 打印中间值"""
    print("\n" + "=" * 80)
    print("详细调试 - 检查中间值")
    print("=" * 80)

    # 使用一个可控的测试用例
    size = 100000
    num_unique = 10000

    print(f"\n输入: size={size:,}, 约 {num_unique:,} unique 值")

    data = torch.randint(0, num_unique, (size,), dtype=torch.int64, device=device)

    # Triton 结果
    triton_out, triton_inverse, triton_counts = _unique2(
        data, sorted=True, return_inverse=True, return_counts=True
    )

    # PyTorch 结果
    torch_out, torch_inverse, torch_counts = torch.unique(
        data, sorted=True, return_inverse=True, return_counts=True
    )

    print("\nTriton 结果:")
    print(f"  unique 值数量: {triton_out.numel()}")
    triton_inv_min = triton_inverse.min().item()
    triton_inv_max = triton_inverse.max().item()
    print(f"  inverse_indices 范围: [{triton_inv_min}, {triton_inv_max}]")
    triton_cnt_min = triton_counts.min().item()
    triton_cnt_max = triton_counts.max().item()
    print(f"  counts 范围: [{triton_cnt_min}, {triton_cnt_max}]")
    print(f"  counts 总和: {triton_counts.sum().item()}")

    print("\nPyTorch 结果:")
    print(f"  unique 值数量: {torch_out.numel()}")
    torch_inv_min = torch_inverse.min().item()
    torch_inv_max = torch_inverse.max().item()
    print(f"  inverse_indices 范围: [{torch_inv_min}, {torch_inv_max}]")
    torch_cnt_min = torch_counts.min().item()
    torch_cnt_max = torch_counts.max().item()
    print(f"  counts 范围: [{torch_cnt_min}, {torch_cnt_max}]")
    print(f"  counts 总和: {torch_counts.sum().item()}")

    # 验证重建
    print("\n验证重建:")
    triton_reconstructed = triton_out[triton_inverse.ravel()]
    torch_reconstructed = torch_out[torch_inverse.ravel()]

    triton_match = torch.equal(triton_reconstructed, data.ravel())
    torch_match = torch.equal(torch_reconstructed, data.ravel())

    print(f"  Triton 重建匹配原始数据: {triton_match}")
    print(f"  PyTorch 重建匹配原始数据: {torch_match}")

    if not triton_match:
        mismatch = torch.where(triton_reconstructed != data.ravel())[0]
        print(f"  Triton 不匹配位置数: {len(mismatch)}")
        if len(mismatch) > 0:
            idx = mismatch[0].item()
            print(f"  第一个不匹配: 位置 {idx}")
            print(f"    原始值: {data.ravel()[idx].item()}")
            print(f"    inverse_index: {triton_inverse.ravel()[idx].item()}")
            print(f"    重建值: {triton_reconstructed[idx].item()}")


if __name__ == "__main__":
    import sys

    # 检测设备
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA 不可用，请确保在 GPU 环境下运行")
        sys.exit(1)

    print(f"使用设备: {device}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")

    # 运行测试
    test_specific_path(device)
    test_various_sizes(device)
    test_edge_cases(device)
    test_int32_overflow_specific(device)
    detailed_debug(device)

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
