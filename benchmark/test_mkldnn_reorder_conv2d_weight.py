# Copyright 2026, The FlagGems Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark for ``torch.ops.aten.mkldnn_reorder_conv2d_weight``.

One case is one conv2d weight geometry plus its convolution attributes. The operator is a CPU
oneDNN weight re-layout, so both the reference and the candidate receive the opaque mkldnn
weight produced by ``to_mkldnn()``; the weight is moved to the CPU first, because that layout
only exists there. The weight is only read, so cached inputs are reused.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# Representative conv2d weight geometries (out_channels, in_channels / groups, kh, kw).
DEFAULT_WEIGHT_SHAPES = [
    (32, 16, 3, 3),
    (64, 64, 3, 3),
    (128, 256, 3, 3),
    (256, 512, 3, 3),
]

# The complete native CPU dtype matrix for this reorder, verified with a valid rank-4 weight.
_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int8]

_PADDING = [1, 1]
_STRIDE = [1, 1]
_DILATION = [1, 1]
_GROUPS = 1


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"weight": shape},
        params={
            "padding": list(_PADDING),
            "stride": list(_STRIDE),
            "dilation": list(_DILATION),
            "groups": _GROUPS,
        },
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    dense = torch.empty(shape, dtype=dtype)
    # Flat positional argument plus a trailing kwargs dict, which is how the harness unpacks
    # a benchmark input; the mkldnn weight is the one positional argument of the operator.
    return dense.to_mkldnn(), dict(plan.params)


class MkldnnReorderConv2dWeightBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        shapes = [tuple(shape) for shape in list(self.shapes) + DEFAULT_WEIGHT_SHAPES]
        # Native descriptors reject rank-one and rank-two weights.
        self.shapes = list(dict.fromkeys(shape for shape in shapes if len(shape) >= 3))


@pytest.mark.mkldnn_reorder_conv2d_weight
def test_mkldnn_reorder_conv2d_weight():
    bench = MkldnnReorderConv2dWeightBenchmark(
        op_name="mkldnn_reorder_conv2d_weight",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_reorder_conv2d_weight,
        gems_op=getattr(flag_gems, "mkldnn_reorder_conv2d_weight", None),
        dtypes=_DTYPES,
    )
    bench.run()
