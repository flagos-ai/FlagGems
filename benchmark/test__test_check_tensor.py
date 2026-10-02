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

# This bool-only self-test accepts any shape, so the shared core/comprehensive shapes are
# kept: set_shapes runs the base implementation and then unions a normalized, de-duplicated
# copy of the extra shapes below. A caller-supplied shape file still replaces the shared
# defaults.
LOCAL_SHAPES = [
    (2, 19, 7),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


def _as_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def _dedup(shapes):
    merged = []
    seen = set()
    for shape in shapes:
        shape = _as_shape(shape)
        if shape not in seen:
            seen.add(shape)
            merged.append(shape)
    return merged


def _storage_shape(shape, layout):
    if layout == "transposed":
        return tuple(shape[:-2]) + (shape[-1], shape[-2])
    if layout == "expanded":
        return (1,) + tuple(shape[1:])
    return tuple(shape)


def _build_view(base_tensor, shape, layout):
    if layout == "transposed":
        return base_tensor.transpose(-1, -2)
    if layout == "expanded":
        return base_tensor.expand(tuple(shape))
    return base_tensor


def _plans_for_shape(shape):
    # One plan per layout: the plain read plus the transposed and stride-0 expanded
    # reads of the same all-true operand.
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"layout": "contiguous"},
        builder_args=(shape, "contiguous"),
    )
    if len(shape) >= 2:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"layout": "transposed"},
            builder_args=(shape, "transposed"),
        )
    if len(shape) >= 1 and shape[0] > 1:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"layout": "expanded"},
            builder_args=(shape, "expanded"),
        )


def _case_fn(shape, dtype):
    del dtype
    yield from _plans_for_shape(_as_shape(shape))


def _build_inputs_fn(plan, dtype, device):
    shape, layout = plan.builder_args
    # The operator requires an all-true operand, so every element is written: no
    # uninitialized storage ever reaches the measurement.
    base_tensor = torch.ones(_storage_shape(shape, layout), dtype=dtype, device=device)
    return _build_view(base_tensor, shape, layout), {}


class CheckTensorBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = _dedup(list(self.shapes) + LOCAL_SHAPES)


@pytest.mark.test_check_tensor
def test__test_check_tensor():
    bench = CheckTensorBenchmark(
        op_name="_test_check_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._test_check_tensor,
        gems_op=getattr(flag_gems, "_test_check_tensor", None),
        dtypes=consts.BOOL_DTYPES,
    )
    bench.run()
