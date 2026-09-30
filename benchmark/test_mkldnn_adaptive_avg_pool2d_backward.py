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
#
# Performance tests for aten::mkldnn_adaptive_avg_pool2d_backward.
#
# The native kernel only serves 4-D MkldnnCPU operands, i.e. host oneDNN
# tensors, so the grid is filtered to that rank and the inputs are built on the
# host device.

import pytest
import torch

import flag_gems

from . import base

# Native-valid dense families for to_mkldnn() (probed).
BENCH_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# Probed-valid 4-D geometries. Even spatial extents give a 2x2 average window;
# (2, 3, 8, 6) keeps an asymmetric 2x3 window and (16, 7, 58, 32) an odd
# batch/channel pair with even spatial extents.
EXTRA_INPUT_SHAPES = [
    (16, 128, 64, 60),
    (16, 32, 64, 60),
    (16, 7, 58, 32),
    (4, 8, 32, 32),
    (2, 3, 8, 6),
    (1, 3, 32, 32),
]


def _grad_output_shape(input_shape):
    """Derive a native-legal grad_output shape for the 2x2 average window."""
    n, c, h, w = input_shape
    if h % 2 == 0 and w % 2 == 0:
        return (n, c, h // 2, w // 2)
    # Keep a valid identity window for odd extents.
    return input_shape


def _case_fn(shape, dtype):
    del dtype
    input_shape = tuple(shape)
    grad_shape = _grad_output_shape(input_shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(input_shape), "grad_output": list(grad_shape)},
        params={
            "kernel_size": [
                input_shape[-2] // grad_shape[-2],
                input_shape[-1] // grad_shape[-1],
            ]
        },
        builder_args=(input_shape, grad_shape),
    )


def _build_inputs_fn(plan, dtype, device):
    # MkldnnCPU tensors exist only on the host, so allocate them there.
    del device
    input_shape, grad_shape = plan.builder_args
    grad = torch.empty(grad_shape, dtype=dtype).to_mkldnn()
    inp = torch.empty(input_shape, dtype=dtype).to_mkldnn()
    return grad, inp


class MkldnnAdaptiveAvgPool2dBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Ordinary resolution first, so the shared grid (or a caller
        # --shape_file) is kept; then drop the ranks the native kernel cannot
        # serve and union the probed-valid 4-D geometries in.
        super().set_shapes(shape_file_path)
        kept = [tuple(shape) for shape in self.shapes if len(tuple(shape)) == 4]
        self.shapes = list(dict.fromkeys(kept + EXTRA_INPUT_SHAPES))


@pytest.mark.mkldnn_adaptive_avg_pool2d_backward
def test_mkldnn_adaptive_avg_pool2d_backward():
    bench = MkldnnAdaptiveAvgPool2dBackwardBenchmark(
        op_name="mkldnn_adaptive_avg_pool2d_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_adaptive_avg_pool2d_backward,
        gems_op=getattr(flag_gems, "mkldnn_adaptive_avg_pool2d_backward", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
