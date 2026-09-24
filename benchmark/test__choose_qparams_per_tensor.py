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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# ``_choose_qparams_per_tensor`` starts with an underscore and ``pytest.mark``
# refuses attribute access for such names, so register the marker directly on
# the MarkGenerator: ``@pytest.mark._choose_qparams_per_tensor`` and
# ``-m _choose_qparams_per_tensor`` then both work.
setattr(
    pytest.mark,
    "_choose_qparams_per_tensor",
    MarkDecorator(
        Mark("_choose_qparams_per_tensor", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

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


class ChooseQParamsPerTensorBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=CQPT_SHAPES)

    def set_more_shapes(self):
        return []


@pytest.mark._choose_qparams_per_tensor
def test__choose_qparams_per_tensor():
    bench = ChooseQParamsPerTensorBenchmark(
        op_name="_choose_qparams_per_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._choose_qparams_per_tensor,
        gems_op=getattr(flag_gems, "_choose_qparams_per_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
