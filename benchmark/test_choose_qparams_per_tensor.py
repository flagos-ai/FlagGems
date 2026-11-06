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

"""Benchmark for ``aten::_choose_qparams_per_tensor``.

The op is a whole-tensor min/max reduction, so there is no ``core_shapes.yaml``
entry and the base class would fall back to ``consts.DEFAULT_SHAPES`` (which
includes a 1-B-element tensor whose allocation cost dominates the
measurement). A modest, allocation-friendly shape set is used instead; each
shape is timed for both ``reduce_range`` values.

``torch_op`` is the ATen reference (the perf comparison baseline) and
``gems_op`` is the FlagGems candidate resolved through ``--override``;
both share the exact same call semantics ``op(input, reduce_range=...)``.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils

# Whole-tensor min/max reduction; the base-class fallback includes a
# 1 B-element tensor whose allocation cost dominates the measurement, so these
# allocation-friendly sizes are pinned and timed for both reduce_range values.
CQPT_SHAPES = [
    (65536,),
    (1_048_576,),
    (4096, 1024),
]


def _case_fn(shape, dtype):
    del dtype
    for reduce_range in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"reduce_range": reduce_range},
            builder_args=(shape,),
        )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    # unpack_to_args_kwargs turns the params dict into call kwargs:
    # op(input, reduce_range=...).
    return inp, {"reduce_range": plan.params["reduce_range"]}


class ChooseQParamsPerTensorBenchmark(base.GenericBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        # core_shapes.yaml has no entry for this whole-tensor min/max
        # reduction and the base-class fallback includes a 1 B-element tensor
        # whose allocation cost dominates the measurement; the modest shape
        # list below is pinned instead.
        del shape_file_path
        self.shapes = list(CQPT_SHAPES)
        self.shape_desc = "M, N"


@pytest.mark.choose_qparams_per_tensor
def test_choose_qparams_per_tensor():
    bench = ChooseQParamsPerTensorBenchmark(
        op_name="choose_qparams_per_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._choose_qparams_per_tensor.default,
        gems_op=flag_gems.choose_qparams_per_tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
