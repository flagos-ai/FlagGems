"""
Performance benchmark for roll operator.

This script benchmarks the roll operation using FlagGems benchmark framework.
"""

import random

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES, INT_DTYPES
from benchmark.performance_utils import GenericBenchmark2DOnly, generate_tensor_input

# Import roll operator directly
from flag_gems.ops import roll
from flag_gems.utils import shape_utils

vendor_name = flag_gems.vendor_name


class RollBenchmark(GenericBenchmark2DOnly):
    """
    Base class for benchmarking roll operations.
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
            (1024, 1024),
            (2048, 2048),
            (512, 512, 512),
            (1024, 1024, 1024),
        ]
        return shapes + additional_shapes


def roll_input_fn(shape, cur_dtype, device):
    """Generate input for roll operation."""
    inp = generate_tensor_input(shape, cur_dtype, device)

    # Test various shifts and dims
    if len(shape) == 1:
        shifts = random.randint(1, max(2, shape[0] // 2))
        yield inp, shifts
    elif len(shape) == 2:
        shifts = random.randint(1, max(2, shape[0] // 2))
        dims = random.choice([0, 1])
        yield inp, shifts, dims
        # Also test tuple shifts
        shifts_tuple = (
            random.randint(1, max(2, shape[0] // 2)),
            random.randint(1, max(2, shape[1] // 2)),
        )
        yield inp, shifts_tuple, (0, 1)
    else:
        # For 3D and higher
        shifts = random.randint(1, max(2, shape[-1] // 2))
        dims = random.choice(range(len(shape)))
        yield inp, shifts, dims


@pytest.mark.roll
@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "roll",
            torch.roll,
            roll_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.roll,
        ),
        pytest.param(
            "roll_int",
            torch.roll,
            roll_input_fn,
            INT_DTYPES,
            marks=pytest.mark.roll,
        ),
    ],
)
def test_perf_roll(op_name, torch_op, input_fn, dtypes):
    """Benchmark roll operation."""
    bench = RollBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
    )
    # Set gems operator explicitly
    bench.set_gems(roll)
    bench.run()


@pytest.mark.roll
def test_perf_roll_various_configs():
    """Benchmark roll with various configurations."""

    def roll_various_input_fn(shape, cur_dtype, device):
        inp = generate_tensor_input(shape, cur_dtype, device)

        if len(shape) >= 2:
            # Test single dimension roll
            yield inp, 10, 0
            # Test multiple dimensions roll
            yield inp, (10, 20), (0, 1)
            # Test negative shifts
            yield inp, -5, -1
            # Test large shift (greater than dimension)
            yield inp, shape[0] + 10, 0

    bench = RollBenchmark(
        input_fn=roll_various_input_fn,
        op_name="roll_various",
        torch_op=torch.roll,
        dtypes=FLOAT_DTYPES,
    )
    # Set gems operator explicitly
    bench.set_gems(roll)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run roll benchmark
    print("\n" + "=" * 60)
    print("ROLL PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_roll("roll", torch.roll, roll_input_fn, FLOAT_DTYPES)

    # Run various configs benchmark
    print("\n" + "=" * 60)
    print("ROLL VARIOUS CONFIGURATIONS BENCHMARK")
    print("=" * 60)
    test_perf_roll_various_configs()
