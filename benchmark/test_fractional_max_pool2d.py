from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


class FractionalMaxPool2dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        # Standard 4D shapes for pooling benchmarks
        shapes_4d = [
            (4, 3, 224, 224),
            (16, 64, 56, 56),
            (32, 128, 28, 28),
            (64, 256, 14, 14),
            (128, 512, 7, 7),
        ]
        for shape in shapes_4d:
            yield from self.input_fn(shape, dtype, self.device)


def fractional_max_pool2d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"kernel_size": 2, "output_size": (shape[2] // 2, shape[3] // 2)}
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        if shape[-2] > 5 and shape[-1] > 5:
            yield inp, {
                "kernel_size": 2,
                "output_size": (shape[2] // 4, shape[3] // 4),
            }


@pytest.mark.fractional_max_pool2d
def test_fractional_max_pool2d():
    bench = FractionalMaxPool2dBenchmark(
        input_fn=fractional_max_pool2d_input_fn,
        op_name="fractional_max_pool2d",
        torch_op=torch.nn.functional.fractional_max_pool2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.fractional_max_pool2d)
    bench.run()


def fractional_max_pool2d_backward_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    inp.requires_grad_(True)
    output_size = (shape[2] // 2, shape[3] // 2)
    output, indices = flag_gems.fractional_max_pool2d(
        inp, kernel_size=2, output_size=output_size
    )
    grad_output = torch.randn_like(output)
    yield grad_output, inp, indices, {"kernel_size": 2, "output_size": output_size}


def torch_fractional_max_pool2d_backward_wrapper(grad_output, input, indices, **kwargs):
    output, _ = torch.nn.functional.fractional_max_pool2d(
        input, return_indices=True, **kwargs
    )
    grad_input = torch.autograd.grad(
        outputs=(output,), inputs=(input,), grad_outputs=(grad_output,)
    )
    return grad_input[0]


@pytest.mark.fractional_max_pool2d_backward
def test_fractional_max_pool2d_backward():
    bench = FractionalMaxPool2dBenchmark(
        input_fn=fractional_max_pool2d_backward_input_fn,
        op_name="fractional_max_pool2d_backward",
        torch_op=torch_fractional_max_pool2d_backward_wrapper,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=False,
    )
    bench.set_gems(flag_gems.fractional_max_pool2d_backward)
    bench.run()
