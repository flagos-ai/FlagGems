import pytest
import torch
import time
import flag_gems
from benchmark.utils import benchmark_forward, get_tflops

# Performance test shapes
PERF_SHAPES = [
    ((1024, 1024), (1024,)),
    ((512, 2048), (2048,)),
    ((32, 128, 1024), (128, 1024)),
    ((16, 64, 512, 1024), (512, 1024)),
]

class TestLayerNormPerf:
    def setup_method(self):
        flag_gems.enable()

    def teardown_method(self):
        flag_gems.disable()

    @pytest.mark.layer_norm
    @pytest.mark.parametrize("shape, normalized_shape", PERF_SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_perf_layer_norm(self, shape, normalized_shape, dtype):
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        weight = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)
        bias = torch.randn(normalized_shape, dtype=dtype, device=flag_gems.device)

        # Warmup
        for _ in range(10):
            _ = flag_gems.experimental.generated_ops.layer_norm(inp, normalized_shape, weight, bias)

        torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(100):
            out = flag_gems.experimental.generated_ops.layer_norm(inp, normalized_shape, weight, bias)
        torch.cuda.synchronize()
        end_time = time.time()

        gems_time = (end_time - start_time) / 100

        # PyTorch baseline
        start_time = time.time()
        for _ in range(100):
            ref_out = torch.layer_norm(inp, normalized_shape, weight, bias)
        torch.cuda.synchronize()
        end_time = time.time()

        torch_time = (end_time - start_time) / 100
        speedup = torch_time / gems_time

        print(f"LayerNorm {shape} {dtype}:")
        print(f"  FlagGems: {gems_time*1000:.3f}ms")
        print(f"  PyTorch: {torch_time*1000:.3f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Assert reasonable speedup
        assert speedup > 1.0, f"LayerNorm should be faster than PyTorch, got {speedup:.2f}x"

