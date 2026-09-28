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

# aten::_reshape_from_tensor(Tensor self, Tensor shape) returns a view of the operand
# when the requested target is stride-expressible from the operand's strides, and
# otherwise materializes a fresh contiguous copy.  Whether the outcome is a view or a
# copy therefore depends on stride compatibility rather than on rank alone; the plans
# below cover both outcomes across these ranks (1-4) and element counts.
RFT_SHAPES = [
    (256,),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    (16, 128, 64, 60),
    (8, 16, 32, 64),
]


def _validated_shape(shape):
    # Called both while listing and while building, so a bad shape file fails loudly
    # instead of producing metadata that does not describe the allocated operand.  A
    # bare integer is a valid 1-D request and is normalized here.
    if isinstance(shape, int) and not isinstance(shape, bool):
        shape = (shape,)
    extents = []
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(
                "benchmark shape needs non-negative integer extents, got {!r}".format(
                    shape
                )
            )
        extents.append(int(extent))
    return tuple(extents)


def _numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _contiguous_strides(shape):
    # torch gives an empty extent the stride of the next non-empty dimension, so e.g.
    # (2, 0, 3) has strides (3, 3, 1).  Listing runs this without any tensor, so the
    # convention is reproduced here rather than read back from an allocation.
    strides = [0] * len(shape)
    tail = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = tail
        tail *= max(shape[axis], 1)
    return strides


def _strided_strides(shape):
    # The builder takes every other row of a doubled first dimension, so the leading
    # stride doubles while the remaining ones stay contiguous.
    strides = _contiguous_strides(shape)
    if strides:
        strides[0] *= 2
    return strides


def _multi_dim_target(shape):
    # A 2-D target that a contiguous operand can still express with strides, so the
    # reshape stays a view while the shape analysis handles a rank change.
    total = _numel(shape)
    if total >= 2 and total % 2 == 0:
        return (total // 2, 2)
    return (1, total) if total else (0,)


def _can_take_strided_operand(shape):
    # A rank-0 tensor has no axis to stride over (slicing raises IndexError) and a
    # zero-extent operand has no element to stride over, so the strided plan is only
    # listed where it has real geometry. Custom rank-0/zero-extent shapes keep the
    # contiguous plans.
    return len(shape) >= 1 and all(extent > 0 for extent in shape)


def _case_fn(shape, dtype):
    del dtype
    shape = _validated_shape(shape)
    # View path: flattening a contiguous operand needs no materialization.
    yield base.BenchmarkCasePlan(
        shape={
            "input": list(shape),
            "operand_strides": _contiguous_strides(shape),
            "target": [-1],
        },
        params={"layout": "contiguous", "target": "flatten"},
        builder_args=(shape, (-1,), "contiguous"),
    )
    # View path with a multi-dim target: stride-expressible shape analysis.
    multi_dim = _multi_dim_target(shape)
    yield base.BenchmarkCasePlan(
        shape={
            "input": list(shape),
            "operand_strides": _contiguous_strides(shape),
            "target": list(multi_dim),
        },
        params={"layout": "contiguous", "target": "multi_dim"},
        builder_args=(shape, multi_dim, "contiguous"),
    )
    # Strided operand of exactly the requested shape, read from a doubled backing
    # buffer.  Whether its flattened target stays a view or materializes a copy is
    # decided by stride compatibility, not by rank: the fixed multidimensional scales
    # here have a leading extent > 1, so that flattened target is not stride-
    # expressible and a copy is produced, while custom shapes whose strided operand
    # still admits a stride-expressible flatten (e.g. singleton dimensions) remain
    # views.
    if _can_take_strided_operand(shape):
        yield base.BenchmarkCasePlan(
            shape={
                "input": list(shape),
                "operand_strides": _strided_strides(shape),
                "target": [-1],
            },
            params={"layout": "strided", "target": "flatten"},
            builder_args=(shape, (-1,), "strided"),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, target, layout = plan.builder_args
    shape = _validated_shape(shape)
    if layout == "strided":
        backing = (2 * shape[0],) + shape[1:]
        operand = utils.generate_tensor_input(backing, dtype, device)[::2]
    else:
        operand = utils.generate_tensor_input(shape, dtype, device)
    # The shape operand is metadata: native requires a 1-D int64 tensor and reads its
    # entries as host values, so it is created here on the CPU.  An accelerator-
    # resident shape tensor is excluded from this file as unverified, based on a prior
    # report rather than on an inspected execution record; no such tensor is built and
    # no runtime probe of that path is performed.
    shape_tensor = torch.tensor(list(target), dtype=torch.long)
    return operand, shape_tensor, {}


def _benchmark_dtypes():
    # The device capability flag decides whether bf16 operands can be allocated; it is
    # read directly so listing and execution agree on the dtype set.
    if flag_gems.runtime.device.support_bf16:
        return list(consts.FLOAT_DTYPES)
    return [dtype for dtype in consts.FLOAT_DTYPES if dtype is not torch.bfloat16]


class ReshapeFromTensorBenchmark(OperatorBenchmark):
    # The generic elementwise shape file targets arithmetic kernels; this operator
    # dispatches only on the shape pair, so RFT_SHAPES is the default while a
    # user-provided --shape_file still takes precedence.

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=RFT_SHAPES)


@pytest.mark.reshape_from_tensor
def test__reshape_from_tensor():
    bench = ReshapeFromTensorBenchmark(
        op_name="_reshape_from_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._reshape_from_tensor,
        gems_op=getattr(flag_gems, "_reshape_from_tensor", None),
        dtypes=_benchmark_dtypes(),
    )
    bench.run()
