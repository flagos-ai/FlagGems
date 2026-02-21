"""Performance benchmarks for smooth_l1_loss operator."""
from typing import Generator

import pytest
import torch

import flag_gems

from .attri_util import FLOAT_DTYPES, BenchLevel
from .performance_utils import Config, GenericBenchmark


def smooth_l1_loss_input_fn(shape, dtype, device):
    """Generate input configurations for smooth_l1_loss benchmarks."""
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)

    # Configuration 1: mean reduction, beta=1.0
    yield inp, target, {
        "reduction": "mean",
        "beta": 1.0,
    }

    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        # Configuration 2: sum reduction
        yield inp, target, {
            "reduction": "sum",
            "beta": 1.0,
        }

        # Configuration 3: none reduction
        yield inp, target, {
            "reduction": "none",
            "beta": 1.0,
        }

        # Configuration 4: different beta
        yield inp, target, {
            "reduction": "mean",
            "beta": 0.5,
        }

        # Configuration 5: different beta
        yield inp, target, {
            "reduction": "mean",
            "beta": 2.0,
        }


class SmoothL1LossBenchmark(GenericBenchmark):
    """Benchmark class for smooth_l1_loss operations."""

    def get_input_iter(self, cur_dtype) -> Generator:
        # Test shapes: small, medium, and large sizes
        shapes = [
            # Small sizes
            (8, 8),  # 64 elements
            (64, 64),  # 4K elements
            # Medium sizes
            (256, 256),  # 64K elements
            (512, 512),  # 256K elements
            (1024, 1024),  # 1M elements
            # Large sizes
            (2048, 2048),  # 4M elements
            (4096, 4096),  # 16M elements
        ]

        for shape in shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.smooth_l1_loss
def test_perf_smooth_l1_loss():
    """Benchmark forward pass of smooth_l1_loss."""
    bench = SmoothL1LossBenchmark(
        input_fn=smooth_l1_loss_input_fn,
        op_name="smooth_l1_loss",
        torch_op=torch.nn.functional.smooth_l1_loss,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.smooth_l1_loss)
    bench.run()


@pytest.mark.smooth_l1_loss
def test_perf_smooth_l1_loss_backward():
    """Benchmark backward pass of smooth_l1_loss."""

    def smooth_l1_loss_backward_input_fn(shape, dtype, device):
        for forward_args in smooth_l1_loss_input_fn(shape, dtype, device):
            inp, target, params = forward_args
            inp.requires_grad_(True)
            output = torch.nn.functional.smooth_l1_loss(inp, target, **params)

            # Create appropriate gradient based on reduction
            if params["reduction"] == "none":
                grad_output = torch.randn_like(output)
            else:
                grad_output = torch.randn((), dtype=dtype, device=device)

            yield grad_output, inp, target, params

    def torch_smooth_l1_loss_backward_wrapper(grad_output, input, target, **kwargs):
        input_copy = input.detach().requires_grad_(True)
        output = torch.nn.functional.smooth_l1_loss(input_copy, target, **kwargs)
        grad_input = torch.autograd.grad(
            outputs=(output,), inputs=(input_copy,), grad_outputs=(grad_output,)
        )
        return grad_input[0]

    bench = SmoothL1LossBenchmark(
        input_fn=smooth_l1_loss_backward_input_fn,
        op_name="smooth_l1_loss_backward",
        torch_op=torch_smooth_l1_loss_backward_wrapper,
        dtypes=FLOAT_DTYPES,
        is_backward=False,
    )

    bench.set_gems(flag_gems.smooth_l1_loss_backward)
    bench.run()
