# benchmark/test_meshgrid_perf.py
import pytest
import torch
import time
from typing import List, Dict
import sys
import os
from tabulate import tabulate

from flag_gems.ops.meshgrid import meshgrid, meshgrid_stack, register_ops


class TestMeshgridPerformance:
    
    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    @pytest.mark.performance
    @pytest.mark.parametrize("size_x, size_y, indexing", [
        (8, 8, "ij"),
        (8, 8, "xy"),
        (256, 256, "ij"),
        (256, 256, "xy"),
        (1024, 1024, "ij"),
        (1024, 1024, "xy"),
    ])
    def test_performance_2d(self, device, size_x, size_y, indexing):
        """Test 2D meshgrid performance"""
        sizes = (size_x, size_y)
        total_elements = size_x * size_y
        
        result = self._benchmark_case(device, '2D', sizes, indexing, total_elements)
        self._print_single_result(result)
        
        assert result['correct'], f"2D meshgrid incorrect for {size_x}x{size_y} with {indexing}"
    
    @pytest.mark.performance
    @pytest.mark.parametrize("size_x, size_y, size_z, indexing", [
        (16, 16, 16, "ij"),
        (16, 16, 16, "xy"),
        (128, 128, 128, "ij"),
        (128, 128, 128, "xy"),
        (512, 512, 512, "ij"),
        (512, 512, 512, "xy"),
    ])
    def test_performance_3d(self, device, size_x, size_y, size_z, indexing):
        """Test 3D meshgrid performance"""
        sizes = (size_x, size_y, size_z)
        total_elements = size_x * size_y * size_z
        
        result = self._benchmark_case(device, '3D', sizes, indexing, total_elements)
        self._print_single_result(result)
        
        assert result['correct'], f"3D meshgrid incorrect for {size_x}x{size_y}x{size_z} with {indexing}"
    
    @pytest.mark.performance
    @pytest.mark.parametrize("size_x, size_y, size_z, size_w, indexing", [
        (8, 8, 8, 8, "ij"),
        (8, 8, 8, 8, "xy"),
        (64, 64, 64, 64, "ij"),
        (64, 64, 64, 64, "xy"),
        (128, 128, 128, 128, "ij"),
        (128, 128, 128, 128, "xy"),
    ])
    def test_performance_4d(self, device, size_x, size_y, size_z, size_w, indexing):
        """Test 4D meshgrid performance"""
        sizes = (size_x, size_y, size_z, size_w)
        total_elements = size_x * size_y * size_z * size_w
        
        result = self._benchmark_case(device, '4D', sizes, indexing, total_elements)
        self._print_single_result(result)
        
        assert result['correct'], f"4D meshgrid incorrect for {size_x}x{size_y}x{size_z}x{size_w} with {indexing}"
    
    @pytest.mark.performance
    @pytest.mark.parametrize("size_x, size_y, indexing", [
        (8, 8, "ij"),
        (8, 8, "xy"),
        (256, 256, "ij"),
        (256, 256, "xy"),
        (1024, 1024, "ij"),
        (1024, 1024, "xy"),
    ])
    def test_performance_2d_tiled(self, device, size_x, size_y, indexing):
        """Test 2D meshgrid performance with tiled kernel"""
        sizes = (size_x, size_y)
        total_elements = size_x * size_y
        
        # Use tiled kernel for larger sizes
        if total_elements > 10000:
            result = self._benchmark_case(device, '2D', sizes, indexing, total_elements, use_tiled=True)
            self._print_single_result(result)
            assert result['correct'], f"2D tiled meshgrid incorrect for {size_x}x{size_y} with {indexing}"
    
    @pytest.mark.benchmark
    def test_compare_all_dimensions(self, device):
        """Comprehensive benchmark comparing all dimensions"""
        test_cases = []
        
        # 2D test cases
        for size_x, size_y in [(8, 8), (256, 256), (1024, 1024)]:
            for indexing in ["ij", "xy"]:
                test_cases.append({
                    'dim': '2D',
                    'sizes': (size_x, size_y),
                    'indexing': indexing,
                    'elements': size_x * size_y
                })
        
        # 3D test cases
        for size_x, size_y, size_z in [(16, 16, 16), (128, 128, 128), (512, 512, 512)]:
            for indexing in ["ij", "xy"]:
                test_cases.append({
                    'dim': '3D',
                    'sizes': (size_x, size_y, size_z),
                    'indexing': indexing,
                    'elements': size_x * size_y * size_z
                })
        
        # 4D test cases
        for size_x, size_y, size_z, size_w in [(8, 8, 8, 8), (64, 64, 64, 64), (128, 128, 128, 128)]:
            for indexing in ["ij", "xy"]:
                test_cases.append({
                    'dim': '4D',
                    'sizes': (size_x, size_y, size_z, size_w),
                    'indexing': indexing,
                    'elements': size_x * size_y * size_z * size_w
                })
        
        results = []
        for case in test_cases:
            result = self._benchmark_case(
                device, 
                case['dim'], 
                case['sizes'], 
                case['indexing'], 
                case['elements']
            )
            results.append(result)
        
        self._print_results_table(results)
        
        for result in results:
            assert result['correct'], f"Benchmark failed for {result['dim']} {result['size']}"
    
    def _benchmark_case(self, device, dim: str, sizes: tuple, indexing: str, total_elements: int, use_tiled: bool = False) -> Dict:
        """Benchmark a single case"""
        tensors = [torch.linspace(0, size, size, device=device) for size in sizes]
        
        # Determine number of iterations based on size
        if total_elements < 10000:
            num_iterations = 100
        elif total_elements < 100000:
            num_iterations = 50
        elif total_elements < 1000000:
            num_iterations = 20
        elif total_elements < 10000000:
            num_iterations = 10
        else:
            num_iterations = 5
        
        # Warmup
        warmup_result = meshgrid(tensors, indexing=indexing)
        _ = sum(r.sum() for r in warmup_result)
        _ = torch.meshgrid(*tensors, indexing=indexing)
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark our implementation
        our_times = []
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            our_result = meshgrid(tensors, indexing=indexing)
            our_sum = sum(r.sum() for r in our_result)
            if device.type == "cuda":
                torch.cuda.synchronize()
            our_times.append(time.perf_counter() - start)
        
        our_times.sort()
        if len(our_times) > 5:
            our_times = our_times[1:-1]
        our_time = sum(our_times) / len(our_times)
        
        # Benchmark PyTorch implementation
        torch_times = []
        for _ in range(num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            torch_result = torch.meshgrid(*tensors, indexing=indexing)
            torch_sum = sum(r.sum() for r in torch_result)
            if device.type == "cuda":
                torch.cuda.synchronize()
            torch_times.append(time.perf_counter() - start)
        
        torch_times.sort()
        if len(torch_times) > 5:
            torch_times = torch_times[1:-1]
        torch_time = sum(torch_times) / len(torch_times)
        
        # Check correctness
        correct = True
        for our, torch_out in zip(our_result, torch_result):
            if not torch.allclose(our, torch_out):
                correct = False
                break
        
        speedup = torch_time / our_time if our_time > 0 else 0
        
        size_str = "x".join(str(s) for s in sizes)
        
        return {
            'dim': dim,
            'size': size_str,
            'elements': total_elements,
            'indexing': indexing.upper(),
            'our_time_ms': our_time * 1000,
            'torch_time_ms': torch_time * 1000,
            'speedup': speedup,
            'correct': correct,
            'num_iterations': num_iterations
        }
    
    def _print_single_result(self, result: Dict):
        """Print a single benchmark result"""
        print(f"\n{result['dim']} {result['size']} ({result['indexing']}):")
        print(f"  Our:     {result['our_time_ms']:.4f} ms")
        print(f"  PyTorch: {result['torch_time_ms']:.4f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")
        print(f"  Correct: {'✓' if result['correct'] else '✗'}")
        print(f"  Iterations: {result['num_iterations']}")
    
    def _print_results_table(self, results: List[Dict]):
        """Print all results in a table"""
        print("\n" + "=" * 130)
        print("MeshGrid Performance Comparison Results")
        print("=" * 130)
        
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
        
        headers = ["Dim", "Size", "Elements", "Indexing", "Our (ms)", "PyTorch (ms)", "Speedup", "Correct"]
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))
        print("=" * 130)


if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "-s",
        "--tb=short"
    ])

