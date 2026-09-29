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

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# aten::cudnn_convolution_add_relu is an NCHW (4-D) / NCDHW (5-D) convolution.
# Each row is one geometry plus the bias form and alpha value to run it with:
#   (input, weight, stride, padding, dilation, groups, bias, alpha)
# The residual always takes the exact convolution output shape, and every bias
# form below is natively valid against that output.
_BENCH_CASES = [
    ((16, 128, 64, 60), (32, 128, 3, 3), (1, 1), (1, 1), (1, 1), 1, "channel_11", None),
    ((16, 128, 64, 60), (16, 128, 3, 3), (1, 1), (1, 1), (1, 1), 1, "channel_11", None),
    ((8, 64, 56, 56), (64, 64, 3, 3), (1, 1), (1, 1), (1, 1), 1, "none", 1.0),
    ((4, 32, 128, 128), (32, 32, 3, 3), (1, 1), (1, 1), (1, 1), 1, "channel", None),
    ((32, 16, 28, 28), (32, 16, 5, 5), (1, 1), (2, 2), (1, 1), 1, "channel_11", 0.5),
    (
        (1, 3, 224, 224),
        (64, 3, 7, 7),
        (2, 2),
        (3, 3),
        (1, 1),
        1,
        "broadcast_1C11",
        None,
    ),
    ((2, 8, 64, 60), (8, 1, 3, 3), (1, 1), (1, 1), (1, 1), 8, "none", None),
    ((16, 128, 64, 60), (32, 128, 3, 3), (1, 1), (1, 1), (1, 1), 1, "none", 1.0),
    ((2, 16, 64, 60), (16, 16, 3, 3), (1, 1), (1, 1), (1, 1), 1, "channel", None),
    ((2, 16, 64, 60), (16, 16, 5, 5), (2, 2), (2, 2), (1, 1), 1, "channel_11", 0.5),
    ((2, 8, 128, 128), (16, 8, 3, 3), (1, 1), (1, 1), (1, 1), 1, "none", None),
    ((1, 256, 16, 16), (64, 256, 1, 1), (1, 1), (0, 0), (1, 1), 1, "channel_11", None),
    ((2, 4, 10, 8), (4, 2, 3, 2), (2, 1), (1, 2), (1, 2), 2, "channel", None),
    ((2, 4, 10, 8), (4, 2, 3, 2), (2, 1), (1, 2), (1, 2), 2, "none", None),
    ((1, 2, 1024, 64), (4, 2, 3, 3), (1, 1), (1, 1), (1, 1), 1, "broadcast_1C11", None),
    ((1, 2, 1024, 64), (4, 2, 3, 3), (1, 1), (1, 1), (1, 1), 1, "channel_11", None),
    (
        (2, 4, 8, 8, 8),
        (8, 2, 3, 3, 3),
        (2, 1, 2),
        (1, 0, 1),
        (1, 2, 1),
        2,
        "none",
        None,
    ),
]

_BIAS_FORMS = ("none", "channel", "channel_11", "broadcast_1C11")

# The native kernel executes these float types; bf16 is gated by the runtime
# capability flag, matching the correctness suite.
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _output_shape(x_shape, w_shape, stride, padding, dilation):
    return (x_shape[0], w_shape[0]) + tuple(
        (x_shape[i + 2] + 2 * padding[i] - dilation[i] * (w_shape[i + 2] - 1) - 1)
        // stride[i]
        + 1
        for i in range(len(stride))
    )


def _case_fn(case, dtype):
    # One descriptor per _BENCH_CASES row (or per matching shape-file entry);
    # this runs during listing, so it only builds metadata.
    del dtype
    x_shape, w_shape, stride, padding, dilation, groups, bias_form, alpha = case
    if bias_form not in _BIAS_FORMS:
        raise ValueError("unknown bias form: " + repr(bias_form))
    out_shape = _output_shape(x_shape, w_shape, stride, padding, dilation)
    yield base.BenchmarkCasePlan(
        shape={
            "input": tuple(x_shape),
            "weight": tuple(w_shape),
            "residual": tuple(out_shape),
        },
        params={
            "stride": list(stride),
            "padding": list(padding),
            "dilation": list(dilation),
            "groups": groups,
            "bias": bias_form,
            "alpha": alpha,
        },
        builder_args=(tuple(x_shape), tuple(w_shape), tuple(out_shape)),
    )


def _build_inputs_fn(plan, dtype, device):
    x_shape, w_shape, out_shape = plan.builder_args
    inp = utils.generate_tensor_input(x_shape, dtype, device)
    weight = utils.generate_tensor_input(w_shape, dtype, device)
    z = utils.generate_tensor_input(out_shape, dtype, device)

    channels = out_shape[1]
    bias_form = plan.params["bias"]
    if bias_form == "none":
        bias = None
    elif bias_form == "channel":
        bias = utils.generate_tensor_input((channels,), dtype, device)
    elif bias_form == "channel_11":
        bias = utils.generate_tensor_input((channels, 1, 1), dtype, device)
    elif bias_form == "broadcast_1C11":
        bias = utils.generate_tensor_input((1, channels, 1, 1), dtype, device)
    else:
        raise ValueError("unknown bias form: " + repr(bias_form))

    # The nine operands are returned flat: unpack_to_args_kwargs treats a nested
    # tuple as a single positional argument.
    return (
        inp,
        weight,
        z,
        plan.params["alpha"],
        bias,
        list(plan.params["stride"]),
        list(plan.params["padding"]),
        list(plan.params["dilation"]),
        plan.params["groups"],
    )


class CudnnConvolutionAddReluBenchmark(OperatorBenchmark):
    # The same descriptors drive listing and execution; these rows apply when
    # the shape file has no entry for this operator.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark.cudnn_convolution_add_relu
def test_cudnn_convolution_add_relu():
    bench = CudnnConvolutionAddReluBenchmark(
        op_name="cudnn_convolution_add_relu",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.cudnn_convolution_add_relu,
        gems_op=getattr(flag_gems, "cudnn_convolution_add_relu", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
