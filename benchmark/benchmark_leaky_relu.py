"""
Performance benchmark for leaky_relu operator.

This script benchmarks the LeakyReLU activation function using FlagGems benchmark framework.
"""

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input

# Import leaky_relu operator directly
from flag_gems.ops import leaky_relu, leaky_relu_

vendor_name = flag_gems.vendor_name


class LeakyReLUBenchmark(Benchmark):
    """
    Base class for benchmarking LeakyReLU operations.
    """

    def set_more_metrics(self):
        return ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            # Test with default negative_slope
            yield inp, 0.01

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


def leaky_relu_input_fn(shape, cur_dtype, device):
    """Generate input for LeakyReLU operation with various slopes."""
    inp = generate_tensor_input(shape, cur_dtype, device)
    # Test with different negative_slope values
    yield inp, 0.01  # default
    yield inp, 0.1  # larger slope
    yield inp, 0.001  # smaller slope


@pytest.mark.leaky_relu
def test_perf_leaky_relu():
    """Benchmark LeakyReLU forward operation with default slope."""
    bench = LeakyReLUBenchmark(
        op_name="leaky_relu",
        torch_op=torch.nn.functional.leaky_relu,
        dtypes=FLOAT_DTYPES,
    )
    # Set gems operator explicitly
    bench.set_gems(leaky_relu)
    bench.run()


@pytest.mark.leaky_relu
def test_perf_leaky_relu_various_slopes():
    """Benchmark LeakyReLU with various negative_slope values."""

    class LeakyReLUSlopeBenchmark(Benchmark):
        def set_more_metrics(self):
            return ["tflops"]

        def set_more_shapes(self):
            special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
            sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
            return special_shapes_2d + sp_shapes_3d

        def get_input_iter(self, cur_dtype):
            for shape in self.shapes:
                inp = generate_tensor_input(shape, cur_dtype, self.device)
                # Test with different slopes
                yield inp, 0.01
                yield inp, 0.1
                yield inp, 0.001

        def get_tflops(self, op, *args, **kwargs):
            shape = list(args[0].shape)
            return torch.tensor(shape).prod().item()

    bench = LeakyReLUSlopeBenchmark(
        op_name="leaky_relu_slopes",
        torch_op=lambda x, slope: torch.nn.functional.leaky_relu(
            x, negative_slope=slope
        ),
        dtypes=FLOAT_DTYPES,
    )
    # Set gems operator explicitly
    bench.set_gems(leaky_relu)
    bench.run()


@pytest.mark.leaky_relu_
def test_perf_leaky_relu_inplace():
    """Benchmark LeakyReLU in-place operation."""
    # PyTorch's inplace leaky_relu
    torch_leaky_relu_inplace = lambda x, slope=0.01: torch.nn.functional.leaky_relu(
        x, negative_slope=slope, inplace=True
    )

    bench = LeakyReLUBenchmark(
        op_name="leaky_relu_",
        torch_op=torch_leaky_relu_inplace,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    # Set gems operator explicitly
    bench.set_gems(leaky_relu_)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run forward benchmark
    print("\n" + "=" * 60)
    print("LEAKY_RELU FORWARD PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_leaky_relu()

    # Run various slopes benchmark
    print("\n" + "=" * 60)
    print("LEAKY_RELU VARIOUS SLOPES BENCHMARK")
    print("=" * 60)
    test_perf_leaky_relu_various_slopes()

    # Run in-place benchmark
    print("\n" + "=" * 60)
    print("LEAKY_RELU IN-PLACE PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_leaky_relu_inplace()
