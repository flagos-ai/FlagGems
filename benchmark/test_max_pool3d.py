from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


def max_pool3d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
    }
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Non-cubic kernel/stride/padding
        if shape[-3] > 5 and shape[-2] > 5 and shape[-1] > 5:
            yield inp, {
                "kernel_size": (2, 3, 3),
                "stride": (1, 2, 2),
                "padding": (0, 1, 1),
                "dilation": 1,
                "ceil_mode": False,
            }
        # With dilation (effective kernel = (3-1)*2+1 = 5, need dim+2*pad >= 5)
        if shape[-3] >= 4 and shape[-2] >= 4 and shape[-1] >= 4:
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


class MaxPool3dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes_5d = [
            (4, 3, 16, 56, 56),
            (8, 64, 8, 28, 28),
            (16, 128, 4, 14, 14),
            (32, 256, 2, 7, 7),
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.max_pool3d
def test_perf_max_pool3d():
    bench = MaxPool3dBenchmark(
        input_fn=max_pool3d_input_fn,
        op_name="max_pool3d",
        torch_op=lambda inp, **kwargs: torch.nn.functional.max_pool3d(
            inp, return_indices=True, **kwargs
        ),
        gems_op=flag_gems.max_pool3d_with_indices,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.max_pool3d
def test_perf_max_pool3d_backward():
    def max_pool3d_backward_input_fn(shape, dtype, device):
        for forward_args in max_pool3d_input_fn(shape, dtype, device):
            inp, params = forward_args
            inp.requires_grad_(True)
            output, indices = flag_gems.max_pool3d_with_indices(inp, **params)
            grad_output = torch.randn_like(output)
            yield grad_output, inp, indices, params

    def torch_max_pool3d_backward_wrapper(grad_output, input, indices, **kwargs):
        output, _ = torch.nn.functional.max_pool3d(input, return_indices=True, **kwargs)
        grad_input = torch.autograd.grad(
            outputs=(output,), inputs=(input,), grad_outputs=(grad_output,)
        )
        return grad_input[0]

    bench = MaxPool3dBenchmark(
        input_fn=max_pool3d_backward_input_fn,
        op_name="max_pool3d_backward",
        torch_op=torch_max_pool3d_backward_wrapper,
        gems_op=flag_gems.max_pool3d_backward,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=False,
    )

    bench.run()
