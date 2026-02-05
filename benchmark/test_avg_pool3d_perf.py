"""Performance benchmarks for avg_pool3d operator."""
from typing import Generator

import pytest
import torch

import flag_gems

from .attri_util import FLOAT_DTYPES, BenchLevel
from .performance_utils import Config, GenericBenchmark


def avg_pool3d_input_fn(shape, dtype, device):
    """Generate input configurations for avg_pool3d benchmarks."""
    inp = torch.randn(shape, dtype=dtype, device=device)

    # Configuration 1: Basic - kernel=3, stride=2, padding=1
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
    }

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        # Configuration 2: Non-cubic kernel/stride/padding
        if shape[-3] > 5 and shape[-2] > 5 and shape[-1] > 5:
            yield inp, {
                "kernel_size": (2, 3, 3),
                "stride": (1, 2, 2),
                "padding": (0, 1, 1),
                "ceil_mode": False,
                "count_include_pad": True,
            }

        # Configuration 3: With ceil_mode
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": True,
            "count_include_pad": True,
        }

        # Configuration 4: count_include_pad=False
        yield inp, {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": False,
            "count_include_pad": False,
        }

        # Configuration 5: No padding
        if shape[-3] >= 4 and shape[-2] >= 4 and shape[-1] >= 4:
            yield inp, {
                "kernel_size": 2,
                "stride": 2,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
            }

        # Configuration 6: Large kernel
        if shape[-3] >= 5 and shape[-2] >= 5 and shape[-1] >= 5:
            yield inp, {
                "kernel_size": 5,
                "stride": 2,
                "padding": 2,
                "ceil_mode": False,
                "count_include_pad": True,
            }

        # Configuration 7: Stride=1 (no downsampling)
        yield inp, {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "ceil_mode": False,
            "count_include_pad": True,
        }


class AvgPool3dBenchmark(GenericBenchmark):
    """Benchmark class for avg_pool3d operations."""

    def get_input_iter(self, cur_dtype) -> Generator:
        # 按照赛题要求：小尺寸、常规尺寸、大尺寸
        shapes_5d = [
            # 小尺寸 (Small)
            (1, 1, 4, 4, 4),  # Minimal 3D input
            (2, 3, 8, 8, 8),  # Small 3D input
            # 常规尺寸 (Medium)
            (2, 8, 16, 16, 16),  # Regular 3D input
            (4, 16, 32, 32, 32),  # Medium 3D input
            (2, 32, 28, 28, 28),  # Typical 3D CNN layer
            # 大尺寸 (Large)
            (1, 64, 14, 14, 14),  # Deeper 3D CNN layer
            (1, 128, 8, 8, 8),  # Late 3D CNN layer
            (1, 32, 64, 64, 64),  # Large 3D volume
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.avg_pool3d
def test_perf_avg_pool3d():
    """Benchmark forward pass of avg_pool3d."""
    bench = AvgPool3dBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d",
        torch_op=torch.nn.functional.avg_pool3d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.avg_pool3d)
    bench.run()


@pytest.mark.avg_pool3d
def test_perf_avg_pool3d_backward():
    """Benchmark backward pass of avg_pool3d."""

    def avg_pool3d_backward_input_fn(shape, dtype, device):
        for forward_args in avg_pool3d_input_fn(shape, dtype, device):
            inp, params = forward_args
            inp.requires_grad_(True)
            output = torch.nn.functional.avg_pool3d(inp, **params)
            grad_output = torch.randn_like(output)
            # Add divisor_override to params for backward
            backward_params = params.copy()
            backward_params["divisor_override"] = params.get("divisor_override", None)
            yield grad_output, inp, backward_params

    def torch_avg_pool3d_backward_wrapper(grad_output, input, **kwargs):
        # Remove divisor_override from kwargs for PyTorch backward (it's only for forward)
        forward_kwargs = {k: v for k, v in kwargs.items() if k != "divisor_override"}
        output = torch.nn.functional.avg_pool3d(input, **forward_kwargs)
        grad_input = torch.autograd.grad(
            outputs=(output,), inputs=(input,), grad_outputs=(grad_output,)
        )
        return grad_input[0]

    bench = AvgPool3dBenchmark(
        input_fn=avg_pool3d_backward_input_fn,
        op_name="avg_pool3d_backward",
        torch_op=torch_avg_pool3d_backward_wrapper,
        dtypes=FLOAT_DTYPES,
        is_backward=False,
    )

    bench.set_gems(flag_gems.avg_pool3d_backward)
    bench.run()
