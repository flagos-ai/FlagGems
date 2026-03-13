"""
Performance benchmark for cosh operator.

This script benchmarks the cosh operation using FlagGems benchmark framework.
"""

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input

# Import cosh operator directly
from flag_gems.ops import cosh, cosh_

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


@pytest.mark.cosh
def test_perf_cosh():
    """Benchmark cosh forward operation."""
    bench = UnaryPointwiseBenchmark(
        op_name="cosh",
        torch_op=torch.cosh,
        dtypes=FLOAT_DTYPES,
    )
    # Set gems operator explicitly
    bench.set_gems(cosh)
    bench.run()


@pytest.mark.cosh_
def test_perf_cosh_inplace():
    """Benchmark cosh in-place operation."""
    # PyTorch doesn't have a native cosh_ in older versions,
    # but we can still benchmark it if available
    try:
        torch_cosh_inplace = torch.cosh_
    except AttributeError:
        pytest.skip("torch.cosh_ not available in this PyTorch version")

    bench = UnaryPointwiseBenchmark(
        op_name="cosh_",
        torch_op=torch_cosh_inplace,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    # Set gems operator explicitly
    bench.set_gems(cosh_)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run forward benchmark
    print("\n" + "=" * 60)
    print("COSH FORWARD PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_cosh()

    # Run in-place benchmark
    print("\n" + "=" * 60)
    print("COSH IN-PLACE PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_cosh_inplace()
