"""
Performance benchmark for gcd operator.

This script benchmarks the greatest common divisor (GCD) operation using FlagGems benchmark framework.
"""

import pytest
import torch

import flag_gems
from benchmark.attri_util import INT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input

# Import gcd operator directly
from flag_gems.ops import gcd

vendor_name = flag_gems.vendor_name


class GcdBenchmark(Benchmark):
    """
    Base class for benchmarking gcd operations.
    """

    def set_more_metrics(self):
        return ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            inp2 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


@pytest.mark.gcd
@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "gcd",
            torch.gcd,
            None,  # Will use get_input_iter from benchmark class
            INT_DTYPES,
            marks=pytest.mark.gcd,
        ),
    ],
)
def test_perf_gcd(op_name, torch_op, input_fn, dtypes):
    """Benchmark gcd operation."""
    bench = GcdBenchmark(
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
    )
    # Set gems operator explicitly
    bench.set_gems(gcd)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run gcd benchmark
    print("\n" + "=" * 60)
    print("GCD PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_gcd("gcd", torch.gcd, None, INT_DTYPES)
