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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# aten::mkldnn_reorder_conv3d_weight dispatches to MkldnnCPU only, so the operand
# is allocated on CPU in its native opaque layout. Operator-specific shapes
# supplement the shared grid, retaining all native-valid configured ranks.
_WEIGHT_SHAPES = [
    (64, 32, 3, 3, 3),
    (128, 128, 3, 3, 3),
    (6, 3, 3, 3, 3),
    (8, 3, 3, 3, 3),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# (padding, stride, dilation, groups, input_size) per weight shape. The oneDNN
# weight descriptor needs kernel dims matching input_size and a group count
# dividing both channel dims, so (20, 320, 15) stays at groups=1.
_WEIGHT_CASES = {
    (64, 32, 3, 3, 3): ((0, 0, 0), (1, 1, 1), (1, 1, 1), 1, None),
    (128, 128, 3, 3, 3): ((1, 1, 1), (1, 1, 1), (1, 1, 1), 1, None),
    (6, 3, 3, 3, 3): ((0, 0, 0), (1, 1, 1), (1, 1, 1), 3, None),
    (8, 3, 3, 3, 3): ((0, 0, 0), (1, 1, 1), (1, 1, 1), 1, (1, 3, 16, 16, 16)),
    (20, 320, 15): ((0, 0, 0), (1, 1, 1), (0, 0, 0), 1, None),
    (16, 128, 64, 60): ((1, 2, 3), (2, 2, 2), (1, 1, 1), 1, None),
}
_DEFAULT_WEIGHT_CASE = ((0, 0, 0), (1, 1, 1), (1, 1, 1), 1, None)

# The .out overload is invocable on this backend with an mkldnn buffer of the
# result shape, so these shapes contribute a second plan.
_OUT_FORM_SHAPES = [(64, 32, 3, 3, 3), (16, 128, 64, 60)]

# Full native-supported CPU dtype family for this op (probed): the float dtypes
# plus int8; uint8 is rejected by the oneDNN conv3d weight descriptor.
_BENCH_DTYPES = list(consts.FLOAT_DTYPES) + [torch.int8]


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    padding, stride, dilation, groups, input_size = _WEIGHT_CASES.get(
        shape, _DEFAULT_WEIGHT_CASE
    )
    params = {
        "padding": list(padding),
        "stride": list(stride),
        "dilation": list(dilation),
        "groups": groups,
        "input_size": None if input_size is None else list(input_size),
    }
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={**params, "form": "default"},
        builder_args=(shape,),
    )
    if shape in _OUT_FORM_SHAPES:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={**params, "form": "out"},
            builder_args=(shape,),
        )


def _build_inputs_fn(plan, dtype, device):
    # Flat positional arguments with a trailing kwargs dict, matching the schema
    # op(weight, padding, stride, dilation, groups[, input_size][, out=...]).
    shape = plan.builder_args[0]
    weight = torch.empty(shape, dtype=dtype).to_mkldnn()
    args = [
        weight,
        plan.params["padding"],
        plan.params["stride"],
        plan.params["dilation"],
        plan.params["groups"],
    ]
    if plan.params["input_size"] is not None:
        args.append(plan.params["input_size"])
    if plan.params["form"] == "out":
        out = torch.empty(shape, dtype=dtype).to_mkldnn()
        return (*args, {"out": out})
    return (*args, {})


class MkldnnReorderConv3dWeightBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # The native weight descriptor rejects ranks below three.
        shapes = [tuple(shape) for shape in list(self.shapes) + _WEIGHT_SHAPES]
        self.shapes = list(dict.fromkeys(shape for shape in shapes if len(shape) >= 3))


@pytest.mark.mkldnn_reorder_conv3d_weight
def test_mkldnn_reorder_conv3d_weight():
    bench = MkldnnReorderConv3dWeightBenchmark(
        op_name="mkldnn_reorder_conv3d_weight",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_reorder_conv3d_weight,
        gems_op=getattr(flag_gems, "mkldnn_reorder_conv3d_weight", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
