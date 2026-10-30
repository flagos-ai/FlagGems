import pytest
import torch

import flag_gems

from . import base, consts, utils


def _fractional_backward_forward_fixture(
    inp, kernel_size, output_size, random_samples=None
):
    if flag_gems.vendor_name != "ascend":
        return flag_gems.fractional_max_pool2d(
            inp,
            kernel_size=kernel_size,
            output_size=output_size,
            _random_samples=random_samples,
        )
    # Build canonical flat indices from the actual input and original samples.
    # Ascend's native frontend also uses a CPU fallback for this forward API.
    if random_samples is None:
        random_samples = torch.rand(
            inp.shape[0], inp.shape[1], 2, dtype=torch.float32, device=inp.device
        )
    output, indices = torch.nn.functional.fractional_max_pool2d(
        inp.detach().to(device="cpu", dtype=torch.float64),
        kernel_size=kernel_size,
        output_size=output_size,
        return_indices=True,
        _random_samples=random_samples.to(device="cpu", dtype=torch.float64),
    )
    return output.to(device=inp.device, dtype=inp.dtype), indices.to(device=inp.device)


class FractionalMaxPool2dBenchmark(base.GenericBenchmark):
    pass


def fractional_max_pool2d_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"kernel_size": 2, "output_size": (shape[2] // 2, shape[3] // 2)},
        builder_args=(shape, 2),
    )
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        if shape[-2] > 5 and shape[-1] > 5:
            yield base.BenchmarkCasePlan(
                shape={"input": list(shape)},
                params={
                    "kernel_size": 2,
                    "output_size": (shape[2] // 4, shape[3] // 4),
                },
                builder_args=(shape, 4),
            )


def fractional_max_pool2d_build_inputs_fn(plan, dtype, device):
    shape, divisor = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {
        "kernel_size": 2,
        "output_size": (shape[2] // divisor, shape[3] // divisor),
    }


@pytest.mark.fractional_max_pool2d
def test_fractional_max_pool2d():
    bench = FractionalMaxPool2dBenchmark(
        case_fn=fractional_max_pool2d_case_fn,
        build_inputs_fn=fractional_max_pool2d_build_inputs_fn,
        op_name="fractional_max_pool2d",
        torch_op=torch.nn.functional.fractional_max_pool2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.fractional_max_pool2d)
    bench.run()


def fractional_max_pool2d_backward_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"kernel_size": 2, "output_size": (shape[2] // 2, shape[3] // 2)},
        builder_args=(shape,),
    )


def fractional_max_pool2d_backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    inp.requires_grad_(True)
    output_size = (shape[2] // 2, shape[3] // 2)
    output, indices = _fractional_backward_forward_fixture(
        inp, kernel_size=2, output_size=output_size
    )
    grad_output = torch.randn_like(output)
    return (
        grad_output,
        inp,
        {
            "kernel_size": 2,
            "output_size": output_size,
            "indices": indices,
        },
    )


def torch_fractional_max_pool2d_backward_wrapper(grad_output, input, **kwargs):
    return torch.ops.aten.fractional_max_pool2d_backward(
        grad_output,
        input,
        kwargs["kernel_size"],
        kwargs["output_size"],
        kwargs["indices"],
    )


@pytest.mark.fractional_max_pool2d_backward
def test_fractional_max_pool2d_backward():
    bench = FractionalMaxPool2dBenchmark(
        case_fn=fractional_max_pool2d_backward_case_fn,
        build_inputs_fn=fractional_max_pool2d_backward_build_inputs_fn,
        op_name="fractional_max_pool2d_backward",
        torch_op=torch_fractional_max_pool2d_backward_wrapper,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.fractional_max_pool2d_backward)
    bench.run()
