from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


class MaxPool2dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes_4d = [
            (4, 3, 224, 224),  # Typical input image size
            (16, 64, 56, 56),  # Early ResNet layer output
            (32, 128, 28, 28),  # Mid ResNet layer output
            (64, 256, 14, 14),  # Later ResNet layer output
            (128, 512, 7, 7),  # Final ResNet layer output
        ]

        for shape in shapes_4d:
            yield from self.input_fn(shape, dtype, self.device)


def max_pool2d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)

    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Non-square kernel/stride/padding
        if shape[-2] > 5 and shape[-1] > 5:
            yield inp, {
                "kernel_size": (3, 5),
                "stride": (2, 1),
                "padding": (1, 2),
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


@pytest.mark.max_pool2d_with_indices
def test_max_pool2d_with_indices():
    bench = MaxPool2dBenchmark(
        op_name="max_pool2d_with_indices",
        input_fn=max_pool2d_input_fn,
        torch_op=torch.nn.functional.max_pool2d_with_indices,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.max_pool2d_with_indices)

    bench.run()


def max_pool2d_backward_input_fn(shape, dtype, device):
    for forward_args in max_pool2d_input_fn(shape, dtype, device):
        inp, params = forward_args
        inp.requires_grad_(True)

        # Use FlagGems forward to produce indices compatible with FlagGems backward
        # Note: FlagGems indices format differs from PyTorch's format
        output, indices = flag_gems.max_pool2d_with_indices(inp, **params)
        grad_output = torch.randn_like(output)
        yield grad_output, inp, indices, params


def torch_max_pool2d_backward_wrapper(grad_output, input, indices, **kwargs):
    # For torch baseline, we use torch forward to get compatible indices
    output, _ = torch.nn.functional.max_pool2d_with_indices(input, **kwargs)
    grad_input = torch.autograd.grad(
        outputs=(output,), inputs=(input,), grad_outputs=(grad_output,)
    )
    return grad_input[0]


@pytest.mark.max_pool2d_backward
def test_max_pool2d_backward():
    bench = MaxPool2dBenchmark(
        input_fn=max_pool2d_backward_input_fn,
        op_name="max_pool2d_backward",
        torch_op=torch_max_pool2d_backward_wrapper,
        dtypes=consts.FLOAT_DTYPES,
        # TODO(Qiming): Double check this !!!
        is_backward=False,
    )

    bench.set_gems(flag_gems.max_pool2d_backward)
    bench.run()
