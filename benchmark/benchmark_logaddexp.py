"""
Performance benchmark for logaddexp operator.

This script benchmarks the log-add-exp operation using FlagGems benchmark framework.
"""

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input

# Import logaddexp operator directly
from flag_gems.ops import logaddexp

vendor_name = flag_gems.vendor_name


class BinaryPointwiseBenchmark(Benchmark):
    """
    Base class for benchmarking binary pointwise operations.
    """

    def set_more_metrics(self):
        return ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp1 = generate_tensor_input(shape, cur_dtype, self.device)
            inp2 = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        shape1 = list(args[0].shape)
        return torch.tensor(shape1).prod().item()


@pytest.mark.logaddexp
def test_perf_logaddexp():
    """Benchmark logaddexp operation."""
    bench = BinaryPointwiseBenchmark(
        op_name="logaddexp",
        torch_op=torch.logaddexp,
        dtypes=FLOAT_DTYPES,
    )
    # Set gems operator explicitly
    bench.set_gems(logaddexp)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run logaddexp benchmark
    print("\n" + "=" * 60)
    print("LOGADDEXP PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_logaddexp()
