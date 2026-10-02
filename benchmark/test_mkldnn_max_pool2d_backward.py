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

"""Benchmark cases for the MkldnnCPU-only ``mkldnn_max_pool2d_backward``.

The operator reads opaque ``torch._mkldnn`` CPU tensors, so the builders
allocate their operands on the CPU and drive a grad-enabled
``mkldnn_max_pool2d`` forward: the backward only runs against the oneDNN
pooling workspace that the training forward stores on its output.  ``input`` is
(N, C, H, W) and ``grad_output`` / ``output`` are the matching forward's
(N, C, H_out, W_out).
"""

import pytest
import torch

import flag_gems

from . import base, consts

# 4-D entries of the shared pool shapes, kept at their original extents.  2-D
# pooling reads (N, C, H, W); the shared benchmark defaults are strided-rank
# pointwise shapes that this operator cannot read.
_SHAPES = [
    (1, 1, 1024, 1024),
    (1, 20, 320, 15),
    (16, 128, 64, 60),
    (112, 57, 32, 29),
    (8, 16, 32, 32),
    (2, 3, 9, 9),
]

_CONFIGS = [
    ([2, 2], [2, 2], [0, 0], [1, 1], False),
    ([2, 2], [2, 2], [0, 0], [1, 1], True),
    ([2, 3], [1, 2], [0, 0], [1, 1], False),
]


def _case_fn(shape, dtype):
    del dtype
    for index, config in enumerate(_CONFIGS):
        kernel_size, stride, padding, dilation, ceil_mode = config
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={
                "kernel_size": list(kernel_size),
                "stride": list(stride),
                "padding": list(padding),
                "dilation": list(dilation),
                "ceil_mode": ceil_mode,
            },
            builder_args=(shape, index),
        )


def _build_inputs_fn(plan, dtype, device):
    # MkldnnCPU dispatch: allocate on the CPU, not on the benchmark device.
    del device
    shape, config_index = plan.builder_args
    kernel_size, stride, padding, dilation, ceil_mode = _CONFIGS[config_index]

    dense = torch.testing.make_tensor(shape, dtype=dtype, device="cpu", low=-1, high=1)
    inp = dense.to_mkldnn().requires_grad_(True)
    with torch.enable_grad():
        output = torch.ops.aten.mkldnn_max_pool2d(
            inp, kernel_size, stride, padding, dilation, ceil_mode
        )
    grad_output = torch.testing.make_tensor(
        tuple(output.shape), dtype=dtype, device="cpu", low=-1, high=1
    ).to_mkldnn()
    # Flat positional arguments in the exact native schema order.
    return (
        grad_output,
        output,
        inp.detach(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )


class MkldnnMaxPool2dBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        pool_shapes = [tuple(shape) for shape in self.shapes if len(shape) == 4]
        self.shapes = list(dict.fromkeys(pool_shapes + _SHAPES))


@pytest.mark.mkldnn_max_pool2d_backward
def test_mkldnn_max_pool2d_backward():
    # is_backward stays False: that mode clones every floating operand and
    # re-requires grad on it, which drops the oneDNN pooling workspace.
    bench = MkldnnMaxPool2dBackwardBenchmark(
        op_name="mkldnn_max_pool2d_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_max_pool2d_backward,
        gems_op=getattr(flag_gems, "mkldnn_max_pool2d_backward", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
