# benchmark/test_meshgrid_perf.py
import pytest
import torch
import time
from typing import List, Dict
import sys
import os
from tabulate import tabulate

# 从 flag_gems 导入
from flag_gems.ops.meshgrid import meshgrid, meshgrid_stack, register_ops


class TestMeshgridPerformance:
    """meshgrid 实现性能测试套件"""
    
    @pytest.fixture
    def device(self):
        """检测可用设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def test_compare_with_torch_comprehensive(self, device):
        """全面对比测试：包含所有指定案例和索引模式"""
        # 定义测试案例
        test_cases = []
        
        # 2D 案例: 8x8, 256x256, 1024x1024
        for size_x, size_y in [(8, 8), (256, 256), (1024, 1024)]:
            for indexing in ["ij", "xy"]:
                test_cases.append({
                    'dim': '2D',
                    'sizes': (size_x, size_y),
                    'indexing': indexing,
                    'elements': size_x * size_y
                })
        
        # 3D 案例: 16x16x16, 128x128x128, 512x512x512
        for size_x, size_y, size_z in [(16, 16, 16), (128, 128, 128), (512, 512, 512)]:
            for indexing in ["ij", "xy"]:
                test_cases.append({
                    'dim': '3D',
                    'sizes': (size_x, size_y, size_z),
                    'indexing': indexing,
                    'elements': size_x * size_y * size_z
                })
        
        # 4D 案例: 8x8x8x8, 64x64x64x64, 128x128x128x128
        for size_x, size_y, size_z, size_w in [(8, 8, 8, 8), (64, 64, 64, 64), (128, 128, 128, 128)]:
            for indexing in ["ij", "xy"]:
                test_cases.append({
                    'dim': '4D',
                    'sizes': (size_x, size_y, size_z, size_w),
                    'indexing': indexing,
                    'elements': size_x * size_y * size_z * size_w
                })
        
        # 执行所有测试
        results = []
        for case in test_cases:
            result = self._benchmark_case(device, case)
            results.append(result)
        
        # 打印结果表格
        self._print_results_table(results)
        
        # 验证正确性
        for result in results:
            assert result['correct']
    
    def _benchmark_case(self, device, case: Dict) -> Dict:
        """测试单个案例的性能"""
        dim = case['dim']
        sizes = case['sizes']
        indexing = case['indexing']
        total_elements = case['elements']
        
        # 创建输入张量
        tensors = [torch.linspace(0, size, size, device=device) for size in sizes]
        
        # 确定迭代次数 - 根据元素数量调整
        if total_elements < 10000:
            num_iterations = 100
        elif total_elements < 100000:
            num_iterations = 50
        elif total_elements < 1000000:
            num_iterations = 20
        elif total_elements < 10000000:
            num_iterations = 10
        else:
            num_iterations = 5  # 大张量减少迭代
        
        # 预热 - 强制触发实际计算
        warmup_result = meshgrid(tensors, indexing=indexing)
        # 通过求和强制触发计算
        _ = sum(r.sum() for r in warmup_result)
        _ = torch.meshgrid(*tensors, indexing=indexing)
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # 测试我们的实现 - 强制触发实际计算
        our_times = []
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            our_result = meshgrid(tensors, indexing=indexing)
            # 关键修复：通过求和强制触发实际计算
            our_sum = sum(r.sum() for r in our_result)
            if device.type == "cuda":
                torch.cuda.synchronize()
            our_times.append(time.perf_counter() - start)
        
        # 去除异常值
        our_times.sort()
        if len(our_times) > 5:
            our_times = our_times[1:-1]
        our_time = sum(our_times) / len(our_times)
        
        # 测试 PyTorch - 强制触发实际计算
        torch_times = []
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            torch_result = torch.meshgrid(*tensors, indexing=indexing)
            # 关键修复：通过求和强制触发实际计算
            torch_sum = sum(r.sum() for r in torch_result)
            if device.type == "cuda":
                torch.cuda.synchronize()
            torch_times.append(time.perf_counter() - start)
        
        torch_times.sort()
        if len(torch_times) > 5:
            torch_times = torch_times[1:-1]
        torch_time = sum(torch_times) / len(torch_times)
        
        # 验证正确性
        correct = True
        for our, torch_out in zip(our_result, torch_result):
            if not torch.allclose(our, torch_out):
                correct = False
                break
        
        # 计算加速比
        speedup = torch_time / our_time if our_time > 0 else 0
        
        # 构建尺寸字符串
        size_str = "x".join(str(s) for s in sizes)
        
        return {
            'dim': dim,
            'size': size_str,
            'elements': total_elements,
            'indexing': indexing.upper(),
            'our_time_ms': our_time * 1000,
            'torch_time_ms': torch_time * 1000,
            'speedup': speedup,
            'correct': correct
        }
    
    def _print_results_table(self, results: List[Dict]):
        """打印格式化的结果表格"""
        print("\n" + "=" * 130)
        print("MeshGrid 性能对比测试结果")
        print("=" * 130)
        
        # 准备表格数据
        table_data = []
        for r in results:
            table_data.append([
                r['dim'],
                r['size'],
                f"{r['elements']:,}",
                r['indexing'],
                f"{r['our_time_ms']:.4f}",
                f"{r['torch_time_ms']:.4f}",
                f"{r['speedup']:.2f}x",
                "✓" if r['correct'] else "✗"
            ])
        
        # 打印表格
        headers = ["维度", "尺寸", "元素数", "索引", "我们的实现 (ms)", "PyTorch (ms)", "加速比", "正确性"]
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))
        print("=" * 130)


# 主程序入口
if __name__ == "__main__":
    # 运行测试
    pytest.main([
        __file__, 
        "-v", 
        "-s",
        "--tb=short"
    ])
