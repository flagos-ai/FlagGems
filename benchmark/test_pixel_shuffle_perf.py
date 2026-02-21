"""Performance benchmarks for pixel_shuffle operator."""
import pytest
import torch

import flag_gems

from .attri_util import FLOAT_DTYPES, BenchLevel
from .performance_utils import Config, GenericBenchmark


def pixel_shuffle_input_fn(shape, dtype, device):
    """Generate input configurations for pixel_shuffle benchmarks."""
    # shape is (batch, channels, height, width)
    # We'll test with different upscale factors
    inp = torch.randn(shape, dtype=dtype, device=device)

    # Configuration 1: upscale_factor=2
    yield inp, {"upscale_factor": 2}

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        # Configuration 2: upscale_factor=3
        # Adjust channels to be divisible by 9
        batch, c, h, w = shape
        c_adjusted = ((c // 9) + 1) * 9 if c % 9 != 0 else c
        inp_r3 = torch.randn((batch, c_adjusted, h, w), dtype=dtype, device=device)
        yield inp_r3, {"upscale_factor": 3}

        # Configuration 3: upscale_factor=4
        # Adjust channels to be divisible by 16
        c_adjusted = ((c // 16) + 1) * 16 if c % 16 != 0 else c
        inp_r4 = torch.randn((batch, c_adjusted, h, w), dtype=dtype, device=device)
        yield inp_r4, {"upscale_factor": 4}


class PixelShuffleBenchmark(GenericBenchmark):
    """Benchmark class for pixel_shuffle operations."""

    def get_input_iter(self, cur_dtype):
        # According to the competition requirements: Small, Medium, Large
        # For pixel_shuffle, we need channels divisible by r^2
        # Using r=2, so channels must be divisible by 4
        shapes = [
            # Small - channels must be divisible by 4
            (1, 4, 8, 8),  # Minimal
            (1, 16, 16, 16),  # Small
            # Medium
            (1, 64, 64, 64),  # Medium
            (4, 64, 128, 128),  # Medium with batch
            (1, 256, 256, 256),  # Large channels
            # Large
            (1, 64, 512, 512),  # Large spatial
            (1, 256, 1024, 1024),  # Very large
        ]

        for shape in shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.pixel_shuffle
def test_perf_pixel_shuffle():
    """Benchmark pixel_shuffle operation."""
    bench = PixelShuffleBenchmark(
        input_fn=pixel_shuffle_input_fn,
        op_name="pixel_shuffle",
        torch_op=torch.nn.functional.pixel_shuffle,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.pixel_shuffle)
    bench.run()
