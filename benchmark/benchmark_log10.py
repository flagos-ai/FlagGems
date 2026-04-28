"""
Performance benchmark for log10 operator.

This script benchmarks the base-10 logarithm (log10) operation using FlagGems benchmark framework.
"""

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input

# Import log10 operator directly
from flag_gems.ops import log10, log10_

vendor_name = flag_gems.vendor_name


class UnaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking unary pointwise operations.
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
            yield inp,

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


@pytest.mark.log10
def test_perf_log10():
    """Benchmark log10 forward operation."""
    bench = UnaryPointwiseBenchmark(
        op_name="log10",
        torch_op=torch.log10,
        dtypes=FLOAT_DTYPES,
    )
    # Set gems operator explicitly
    bench.set_gems(log10)
    bench.run()


@pytest.mark.log10_
def test_perf_log10_inplace():
    """Benchmark log10 in-place operation."""
    # PyTorch's log10_ is available
    try:
        torch_log10_inplace = torch.log10_
    except AttributeError:
        pytest.skip("torch.log10_ not available in this PyTorch version")

    bench = UnaryPointwiseBenchmark(
        op_name="log10_",
        torch_op=torch_log10_inplace,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    # Set gems operator explicitly
    bench.set_gems(log10_)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run forward benchmark
    print("\n" + "=" * 60)
    print("LOG10 FORWARD PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_log10()

    # Run in-place benchmark
    print("\n" + "=" * 60)
    print("LOG10 IN-PLACE PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_log10_inplace()
