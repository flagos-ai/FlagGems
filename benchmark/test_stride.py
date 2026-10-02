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

# aten::stride reads element strides from tensor metadata, so each measured call
# is O(1) and the shared benchmark shapes only size the tensor whose metadata is
# read. Both callable overloads are measured, plus a transposed view whose
# reported strides differ from a contiguous one.


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"form": "default", "dim": None, "layout": "contiguous"},
        builder_args=(shape, "default", None, "contiguous"),
    )
    if not shape:
        # A scalar tensor has no dimension, so the single-dim form has no valid
        # case for it; its rejection is covered by the correctness test.
        return
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"form": "dim", "dim": 0, "layout": "contiguous"},
        builder_args=(shape, "dim", 0, "contiguous"),
    )
    if len(shape) >= 2:
        # A transposed view keeps the storage but changes the strides.
        swapped = (shape[-1],) + shape[1:-1] + (shape[0],)
        yield base.BenchmarkCasePlan(
            shape={"input": list(swapped)},
            params={"form": "dim", "dim": -1, "layout": "transposed"},
            builder_args=(shape, "dim", -1, "transposed"),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, form, dim, layout = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if layout == "transposed":
        inp = inp.transpose(0, -1)
    if form == "dim":
        # Flat argument tuple: (tensor, int), matching the reference call form.
        return inp, dim
    return (inp,)


@pytest.mark.stride
def test_stride():
    bench = base.GenericBenchmark(
        op_name="stride",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.stride,
        gems_op=getattr(flag_gems, "stride", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
