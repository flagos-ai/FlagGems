"""
Performance benchmark for tril operator.

This script benchmarks the lower triangular (tril) operation using FlagGems benchmark framework.
"""

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark2DOnly, generate_tensor_input

# Import tril operator directly
from flag_gems.ops import tril
from flag_gems.utils import shape_utils

vendor_name = flag_gems.vendor_name


class TrilBenchmark(GenericBenchmark2DOnly):
    """
    Base class for benchmarking tril operations.
    """

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, inp]])
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_shapes(self):
        """Add more shapes for comprehensive testing."""
        shapes = super().set_more_shapes()
        # Add additional shapes for various dimensions
        additional_shapes = [
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (256, 256, 256),
            (512, 512, 512),
        ]
        return shapes + additional_shapes


def tril_input_fn(shape, cur_dtype, device):
    """Generate input for tril operation with various diagonal values."""
    inp = generate_tensor_input(shape, cur_dtype, device)
    # Only test 2D and higher dimensional tensors
    if len(shape) >= 2:
        # Test with different diagonal values
        yield inp, 0  # main diagonal
        yield inp, 1  # first diagonal above
        yield inp, -1  # first diagonal below
        yield inp, 2  # second diagonal above


@pytest.mark.tril
@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "tril",
            torch.tril,
            tril_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.tril,
        ),
    ],
)
def test_perf_tril(op_name, torch_op, input_fn, dtypes):
    """Benchmark tril operation."""
    bench = TrilBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
    )
    # Set gems operator explicitly
    bench.set_gems(tril)
    bench.run()


@pytest.mark.tril
def test_perf_tril_various_diagonals():
    """Benchmark tril with various diagonal configurations."""

    def tril_diagonal_input_fn(shape, cur_dtype, device):
        inp = generate_tensor_input(shape, cur_dtype, device)

        if len(shape) >= 2:
            # Test main diagonal
            yield inp, 0
            # Test diagonals above main
            yield inp, 1
            yield inp, 2
            yield inp, 3
            # Test diagonals below main
            yield inp, -1
            yield inp, -2

    bench = TrilBenchmark(
        input_fn=tril_diagonal_input_fn,
        op_name="tril_diagonals",
        torch_op=torch.tril,
        dtypes=FLOAT_DTYPES,
    )
    # Set gems operator explicitly
    bench.set_gems(tril)
    bench.run()


@pytest.mark.tril
def test_perf_tril_batched():
    """Benchmark tril with batched tensors (3D+)."""

    def tril_batched_input_fn(shape, cur_dtype, device):
        # Only use 3D+ shapes for batched testing
        if len(shape) >= 3:
            inp = generate_tensor_input(shape, cur_dtype, device)
            yield inp, 0
            yield inp, 1
            yield inp, -1

    bench = TrilBenchmark(
        input_fn=tril_batched_input_fn,
        op_name="tril_batched",
        torch_op=torch.tril,
        dtypes=FLOAT_DTYPES,
    )
    # Set gems operator explicitly
    bench.set_gems(tril)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run tril benchmark
    print("\n" + "=" * 60)
    print("TRIL PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_tril("tril", torch.tril, tril_input_fn, FLOAT_DTYPES)

    # Run various diagonals benchmark
    print("\n" + "=" * 60)
    print("TRIL VARIOUS DIAGONALS BENCHMARK")
    print("=" * 60)
    test_perf_tril_various_diagonals()

    # Run batched benchmark
    print("\n" + "=" * 60)
    print("TRIL BATCHED TENSOR BENCHMARK")
    print("=" * 60)
    test_perf_tril_batched()
