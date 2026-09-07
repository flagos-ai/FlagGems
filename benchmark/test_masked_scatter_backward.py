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
from flag_gems.utils import shape_utils

from . import base, consts, utils

# ---------------------------------------------------------------------------
# Shapes are chosen to evenly cover the three performance regimes of the
# underlying masked_select kernel:
#
#   regime          N           kernel path
#   --------------  ----------  ----------------------------------------
#   small           <= 4096     single-pass, 1 block, tl.cumsum
#   boundary        ~4096       single-pass at max BLOCK_SIZE (=4096)
#   medium          4K - 256K   multi-CTA, moderate block count
#   large           > 256K      multi-CTA, full SM utilisation
# ---------------------------------------------------------------------------

# Core shapes: fast pass, one shape from each regime
CORE_SHAPES = [
    (32, 32),  # N = 1024     small, single-pass
    (64, 65),  # N = 4160     just-above-boundary, 1-CTA degenerate
    (128, 128),  # N = 16384    medium, multi-CTA
    (4096, 4096),  # N = 16M      large, GPU-saturating
]

# Comprehensive shapes: finer granularity
MORE_SHAPES = [
    (17, 17),  # N = 289      tiny, single-pass
    (32, 128),  # N = 4096     boundary, max single-pass
    (128, 64),  # N = 8192     above boundary, 2-D
    (256, 64),  # N = 16384    medium, multi-CTA
    (256, 256),  # N = 65536    medium-large
    (1024, 1024),  # N = 1M       large
    (1024, 2048),  # N = 2M       large, skewed
    (1024, 65536),  # N = 67M      very large, skewed
    (10000, 65536),  # N = 655M     GPU saturation
]


class TensorSelectBackwardBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        if flag_gems.vendor_name == "kunlunxin":
            return []
        shapes = super().set_more_shapes()
        shapes = [
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes + MORE_SHAPES


def _input_fn(shape, cur_dtype, device):
    grad_output = utils.generate_tensor_input(shape, cur_dtype, device)
    mask = utils.generate_tensor_input(shape, cur_dtype, device) < 0.3
    sizes = shape

    yield grad_output, mask, sizes


def _get_gbps(bench_fn_args, latency):
    grad_output, mask, sizes = bench_fn_args
    io_amount = sum(
        [shape_utils.size_in_bytes(item) for item in [grad_output, mask, grad_output]]
    )

    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.masked_scatter_backward
def test_masked_scatter_backward():
    bench = TensorSelectBackwardBenchmark(
        op_name="masked_scatter_backward",
        torch_op=torch.ops.aten.masked_scatter_backward,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    # Override default shapes with hand-picked coverage
    bench.shapes = CORE_SHAPES
    bench.run()
