"""Performance benchmarks for max_pool3d operator."""
import pytest
import torch
from typing import Generator

import flag_gems

from .attri_util import FLOAT_DTYPES, BenchLevel
from .performance_utils import Config, GenericBenchmark


def max_pool3d_input_fn(shape, dtype, device):
    """Generate input configurations for max_pool3d benchmarks."""
    inp = torch.randn(shape, dtype=dtype, device=device)
    
    # Basic configuration
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }
    
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        # Non-cubic kernel/stride/padding
        if shape[-3] > 5 and shape[-2] > 5 and shape[-1] > 5:
            yield inp, {
                "kernel_size": (2, 3, 3),
                "stride": (1, 2, 2),
                "padding": (0, 1, 1),
                "dilation": 1,
                "ceil_mode": False,
            }
        # With dilation
        yield inp, {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 2,
            "ceil_mode": False,
        }
        # With ceil_mode
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "ceil_mode": True,
        }


class MaxPool3dBenchmark(GenericBenchmark):
    """Benchmark class for max_pool3d operations."""
    
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes_5d = [
            (2, 3, 16, 16, 16),    # Small 3D input
            (4, 16, 32, 32, 32),   # Medium 3D input
            (2, 32, 28, 28, 28),   # Typical 3D CNN layer
            (1, 64, 14, 14, 14),   # Deeper 3D CNN layer
            (1, 128, 8, 8, 8),     # Late 3D CNN layer
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.max_pool3d
def test_perf_max_pool3d():
    """Benchmark forward pass of max_pool3d."""
    bench = MaxPool3dBenchmark(
        input_fn=max_pool3d_input_fn,
        op_name="max_pool3d",
        torch_op=torch.nn.functional.max_pool3d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.max_pool3d)
    bench.run()


@pytest.mark.max_pool3d
def test_perf_max_pool3d_with_indices():
    """Benchmark forward pass of max_pool3d with indices."""
    def torch_max_pool3d_with_indices(inp, **kwargs):
        return torch.nn.functional.max_pool3d(inp, return_indices=True, **kwargs)
    
    bench = MaxPool3dBenchmark(
        input_fn=max_pool3d_input_fn,
        op_name="max_pool3d_with_indices",
        torch_op=torch_max_pool3d_with_indices,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.max_pool3d_with_indices)
    bench.run()


@pytest.mark.max_pool3d
def test_perf_max_pool3d_backward():
    """Benchmark backward pass of max_pool3d."""
    def max_pool3d_backward_input_fn(shape, dtype, device):
        for forward_args in max_pool3d_input_fn(shape, dtype, device):
            inp, params = forward_args
            inp.requires_grad_(True)
            output, indices = torch.nn.functional.max_pool3d(
                inp, return_indices=True, **params
            )
            grad_output = torch.randn_like(output)
            yield grad_output, inp, indices, params

    def torch_max_pool3d_backward_wrapper(grad_output, input, indices, **kwargs):
        output, _ = torch.nn.functional.max_pool3d(
            input, return_indices=True, **kwargs
        )
        grad_input = torch.autograd.grad(
            outputs=(output,), inputs=(input,), grad_outputs=(grad_output,)
        )
        return grad_input[0]

    bench = MaxPool3dBenchmark(
        input_fn=max_pool3d_backward_input_fn,
        op_name="max_pool3d_backward",
        torch_op=torch_max_pool3d_backward_wrapper,
        dtypes=FLOAT_DTYPES,
        is_backward=False,
    )

    bench.set_gems(flag_gems.max_pool3d_backward)
    bench.run()
