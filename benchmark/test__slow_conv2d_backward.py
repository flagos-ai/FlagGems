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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# ``_slow_conv2d_backward`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register the
# marker directly on the MarkGenerator so ``@pytest.mark._slow_conv2d_backward``
# and ``-m _slow_conv2d_backward`` both work.
setattr(
    pytest.mark,
    "_slow_conv2d_backward",
    MarkDecorator(
        Mark("_slow_conv2d_backward", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_slow_conv2d_backward(grad_output, self, weight, kernel_size, stride,
# padding, output_mask) -> (grad_input, grad_weight, grad_bias). ``self`` is
# (N, C_in, H, W), ``weight`` is (C_out, C_in, kH, kW) and ``grad_output`` is
# (N, C_out, H_out, W_out) with
#   H_out = (H + 2*pH - kH) // sH + 1, W_out = (W + 2*pW - kW) // sW + 1.
# The benchmark drives the masked ``.output_mask`` overload with an all-true
# mask, i.e. the full-workload shape of the op (all three gradients), and both
# the torch reference and the candidate go through the exact same call.
#
# Shapes are (N, C_in, H, W, C_out, kH, kW, stride, padding) tuples. They span
# 1x1/2x2/3x3/3x5 kernels, stride 1 and 2, padding 0/1/2 and channel counts up
# to 64 (roughly 4K - 4M input elements) so the measurement is
# performance-relevant.
_SLOW_CONV2D_BACKWARD_SHAPES = [
    (16, 4, 8, 8, 4, 3, 3, 1, 0),
    (8, 3, 16, 16, 8, 3, 3, 1, 1),
    (32, 8, 8, 8, 32, 2, 2, 2, 0),
    (32, 8, 8, 8, 32, 2, 2, 1, 1),
    (4, 16, 4, 4, 16, 1, 1, 1, 0),
    (4, 16, 4, 4, 16, 1, 1, 2, 0),
    (2, 3, 9, 9, 4, 3, 5, 1, 2),
    (2, 3, 4, 4, 5, 3, 3, 1, 0),
    (64, 32, 8, 8, 64, 3, 3, 1, 1),
    (8, 64, 16, 16, 64, 3, 3, 2, 1),
]

_FULL_MASK = (True, True, True)


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"output_mask": _FULL_MASK},
        builder_args=(shape, 0),
    )


def _build_inputs_fn(plan, dtype, device):
    n, c_in, h, w, c_out, k_h, k_w, stride, padding = plan.builder_args[0]
    h_out = (h + 2 * padding - k_h) // stride + 1
    w_out = (w + 2 * padding - k_w) // stride + 1

    grad_output = utils.generate_tensor_input((n, c_out, h_out, w_out), dtype, device)
    inp = utils.generate_tensor_input((n, c_in, h, w), dtype, device)
    weight = utils.generate_tensor_input((c_out, c_in, k_h, k_w), dtype, device)
    # Positional args shared by the torch reference and the candidate.
    return (
        grad_output,
        inp,
        weight,
        (k_h, k_w),
        (stride, stride),
        (padding, padding),
        plan.params["output_mask"],
    )


class SlowConv2dBackwardBenchmark(OperatorBenchmark):
    # _slow_conv2d_backward has no entry in benchmark/core_shapes.yaml (its
    # inputs are conv-shaped, not the generic pointwise shapes), so benchmark
    # dedicated (N, C_in, H, W, C_out, kH, kW, stride, padding) tuples instead.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_SLOW_CONV2D_BACKWARD_SHAPES)


@pytest.mark._slow_conv2d_backward
def test__slow_conv2d_backward():
    bench = SlowConv2dBackwardBenchmark(
        op_name="_slow_conv2d_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._slow_conv2d_backward.output_mask,
        gems_op=getattr(flag_gems, "_slow_conv2d_backward", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
