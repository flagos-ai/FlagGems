# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


_MAX_POOL2D_SHAPES_4D = [
    (4, 3, 224, 224),  # Typical input image size
    (16, 64, 56, 56),  # Early ResNet layer output
    (32, 128, 28, 28),  # Mid ResNet layer output
    (64, 256, 14, 14),  # Later ResNet layer output
    (128, 512, 7, 7),  # Final ResNet layer output
]


def _max_pool2d_param_variants(shape):
    # Pure enumeration of the pooling-kwargs variants for one shape, mirroring
    # max_pool2d_input_fn's control flow without allocating any tensors.
    variants = [
        {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "ceil_mode": False,
        }
    ]
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Non-square kernel/stride/padding
        if shape[-2] > 5 and shape[-1] > 5:
            variants.append(
                {
                    "kernel_size": (3, 5),
                    "stride": (2, 1),
                    "padding": (1, 2),
                    "dilation": 1,
                    "ceil_mode": False,
                }
            )

        # With dilation
        variants.append(
            {
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dilation": 2,
                "ceil_mode": False,
            }
        )

        # With ceil_mode
        variants.append(
            {
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "ceil_mode": True,
            }
        )
    return variants


class MaxPool2dCaseBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        if not base.Config.query:
            self.shapes = list(_MAX_POOL2D_SHAPES_4D)


def max_pool2d_case_fn(shape, dtype):
    for params in _max_pool2d_param_variants(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params=params,
            builder_args=(shape,),
        )


def build_max_pool2d_inputs(plan, dtype, device):
    (shape,) = plan.builder_args
    return utils.generate_tensor_input(shape, dtype, device), dict(plan.params)


def build_max_pool2d_backward_inputs(plan, dtype, device):
    inp, params = build_max_pool2d_inputs(plan, dtype, device)
    inp.requires_grad_(True)

    # Preserve the index format consumed by the FlagGems backward.
    output, indices = flag_gems.max_pool2d_with_indices(inp, **params)
    grad_output = torch.randn_like(output)
    return grad_output, inp, indices, params


@pytest.mark.max_pool2d_with_indices
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_max_pool2d_with_indices():
    bench = MaxPool2dCaseBenchmark(
        op_name="max_pool2d_with_indices",
        case_fn=max_pool2d_case_fn,
        build_inputs_fn=build_max_pool2d_inputs,
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
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_max_pool2d_backward():
    bench = MaxPool2dCaseBenchmark(
        case_fn=max_pool2d_case_fn,
        build_inputs_fn=build_max_pool2d_backward_inputs,
        op_name="max_pool2d_backward",
        torch_op=torch_max_pool2d_backward_wrapper,
        dtypes=consts.FLOAT_DTYPES,
        # TODO(Qiming): Double check this !!!
        is_backward=False,
    )

    bench.set_gems(
        lambda *args, **kwargs: flag_gems.max_pool2d_backward(*args, **kwargs)
    )
    bench.run()


def max_pool2d_with_indices_backward_input_fn(shape, dtype, device):
    for forward_args in max_pool2d_input_fn(shape, dtype, device):
        inp, params = forward_args
        # Use FlagGems forward to produce indices compatible with FlagGems backward
        # Note: FlagGems indices format differs from PyTorch's format
        output, indices = flag_gems.max_pool2d_with_indices(inp, **params)
        grad_output = torch.randn_like(output)
        yield (
            grad_output,
            inp,
            params["kernel_size"],
            params["stride"],
            params["padding"],
            params["dilation"],
            params["ceil_mode"],
            indices,
        )


def torch_max_pool2d_with_indices_backward_wrapper(
    grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    # For torch baseline, recompute forward with torch to get compatible indices,
    # then call aten::max_pool2d_with_indices_backward directly
    output, torch_indices = torch.ops.aten.max_pool2d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode
    )
    return torch.ops.aten.max_pool2d_with_indices_backward(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        torch_indices,
    )


@pytest.mark.max_pool2d_with_indices_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_max_pool2d_with_indices_backward():
    bench = MaxPool2dBenchmark(
        input_fn=max_pool2d_with_indices_backward_input_fn,
        op_name="max_pool2d_with_indices_backward",
        torch_op=torch_max_pool2d_with_indices_backward_wrapper,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=False,
    )

    bench.set_gems(flag_gems.max_pool2d_with_indices_backward)
    bench.run()
