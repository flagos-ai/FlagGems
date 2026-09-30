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

from . import base
from .generated_operator_utils import OperatorBenchmark

# Host-only op: the operand must carry the mkldnn layout, only 4-D inputs are
# legal, and the conversion runs on the host (no accelerator path). float64,
# int32, int64, bool and float8 cannot form an operand.
_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
]

_MKLDNN_SHAPES = [
    (2, 3, 8, 8),
    (1, 1, 256, 256),
    (16, 128, 64, 60),
    (20, 320, 15, 16),
    (1, 64, 224, 224),
    (4, 128, 112, 112),
]

# output_size must divide both spatial extents: core lists one dividing size per
# input, comprehensive adds the global-pool and near-identity boundaries.
_CORE_OUTPUT_SIZES = {
    (2, 3, 8, 8): (2, 2),
    (1, 1, 256, 256): (16, 16),
    (16, 128, 64, 60): (2, 2),
    (20, 320, 15, 16): (5, 8),
    (1, 64, 224, 224): (7, 7),
    (4, 128, 112, 112): (14, 14),
}
_EXTRA_OUTPUT_SIZES = {
    (2, 3, 8, 8): ((1, 1), (8, 8)),
    (1, 1, 256, 256): ((1, 1), (32, 32)),
    (16, 128, 64, 60): ((1, 1), (4, 4)),
    (20, 320, 15, 16): ((1, 1), (15, 16)),
    (1, 64, 224, 224): ((1, 1), (14, 14)),
    (4, 128, 112, 112): ((1, 1), (28, 28)),
}


def _output_sizes(shape):
    # A caller-supplied 4-D shape keeps its own geometry; (1, 1) is valid for any
    # non-empty 4-D input, so no requested workload is filtered or replaced.
    key = tuple(shape)
    if key not in _CORE_OUTPUT_SIZES:
        return [(1, 1)]
    sizes = [_CORE_OUTPUT_SIZES[key]]
    sizes.extend(_EXTRA_OUTPUT_SIZES.get(key, ()))
    return sizes


def _case_fn(shape, dtype):
    del dtype
    # Listing and execution share these plans; nothing is allocated here.
    for output_size in _output_sizes(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": tuple(shape)},
            params={"output_size": list(output_size)},
            builder_args=(tuple(shape), tuple(output_size)),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, output_size = plan.builder_args
    inp = torch.empty(shape, dtype=dtype).to_mkldnn()
    return inp, [output_size[0], output_size[1]]


class MkldnnAdaptiveAvgPool2dBenchmark(OperatorBenchmark):
    """Two-phase benchmark restricted to the 4-D mkldnn operand contract."""

    def set_shapes(self, shape_file_path=None):
        # Shared grid and explicit caller configuration first, then native-valid
        # 4-D extras; only illegal ranks are dropped, no geometry is rewritten.
        super().set_shapes(shape_file_path)
        four_d = [tuple(shape) for shape in self.shapes if len(tuple(shape)) == 4]
        self.shapes = list(dict.fromkeys(four_d + _MKLDNN_SHAPES))


@pytest.mark.mkldnn_adaptive_avg_pool2d
def test_mkldnn_adaptive_avg_pool2d():
    bench = MkldnnAdaptiveAvgPool2dBenchmark(
        op_name="mkldnn_adaptive_avg_pool2d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_adaptive_avg_pool2d,
        gems_op=getattr(flag_gems, "mkldnn_adaptive_avg_pool2d", None),
        dtypes=_DTYPES,
    )
    bench.run()
