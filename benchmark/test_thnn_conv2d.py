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

"""Benchmark for ``aten::thnn_conv2d``.

The op is the legacy THNN 2-D convolution over NCHW operands: it does not fit
the pointwise or BLAS benchmark families, so a dedicated Benchmark subclass
carries its (shape, out_channels, kernel, padding, stride) cases. The timed
grid covers the 1x1 pure-GEMM path and the 3x3 im2col path on sizes that stay
allocation-friendly.
"""

import pytest
import torch

import flag_gems

from . import base, consts

# NCHW shapes: the first extent is the batch. The 1x1 row exercises the
# pure-GEMM path and the 3x3 rows the deep-K im2col path.
CONV_SHAPES = [
    (2, 3, 16, 16),
    (1, 16, 32, 32),
    (4, 8, 9, 7),
    (8, 4, 28, 28),
    (16, 32, 16, 16),
]

# (out_channels multiplier, kernel, padding, stride) variants. The out-channel
# count is derived from in_channels so every shape stays legal.
_VARIANTS = [
    (2, 3, [1, 1], [1, 1]),
    (2, 1, [0, 0], [1, 1]),
    (1, 3, [1, 1], [2, 2]),
]


def _case_fn(shape, dtype):
    del dtype
    for multiplier, kernel, padding, stride in _VARIANTS:
        out_channels = min(64, max(4, shape[1] * multiplier))
        yield consts.BenchmarkCasePlan(
            shape={"input": list(shape), "out": [shape[0], out_channels]},
            params={
                "out_channels": out_channels,
                "kernel": kernel,
                "padding": str(padding),
                "stride": str(stride),
            },
            builder_args=(shape, out_channels, kernel, padding, stride),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, out_channels, kernel, padding, stride = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(
        (out_channels, shape[1], kernel, kernel), dtype=dtype, device=device
    )
    bias = torch.randn(out_channels, dtype=dtype, device=device)
    # Positional order matches the aten schema: (self, weight, kernel_size,
    # bias, stride, padding).
    return inp, weight, (kernel, kernel), bias, stride, padding


class ThnnConv2dBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # core_shapes.yaml has no entry for this legacy convolution and the
        # base-class fallback is a plain pointwise shape list; the curated NCHW
        # list above is pinned instead.
        del shape_file_path
        self.shapes = list(CONV_SHAPES)
        self.shape_desc = "N, C, H, W"


@pytest.mark.thnn_conv2d
def test_thnn_conv2d():
    bench = ThnnConv2dBenchmark(
        op_name="thnn_conv2d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.thnn_conv2d.default,
        gems_op=flag_gems.thnn_conv2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
