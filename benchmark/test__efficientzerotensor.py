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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts

# ``_efficientzerotensor`` starts with an underscore, and ``pytest.mark``
# refuses to create a marker through attribute access for such names. Register
# the markers on the MarkGenerator directly so both
# ``@pytest.mark._efficientzerotensor`` and ``-m _efficientzerotensor`` work.
for _name in ("_efficientzerotensor", "_efficientzerotensor_out"):
    setattr(
        pytest.mark,
        _name,
        MarkDecorator(Mark(_name, (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_efficientzerotensor is a factory that allocates a real, zero-filled
# tensor on the active device. The default shape set contains a 1-B-element
# 1-D tensor whose cost would be dominated by allocation rather than by the
# measured call, and the allocation-friendly 4-D entry keeps the fill work
# bounded; both variants below therefore use this local shape list instead of
# resolving core_shapes.yaml (which has no entry for this operator).
EFFICIENTZEROTENSOR_SHAPES = [
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (20, 320, 15),
    (16, 128, 64, 1280),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"size": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    # Factory form: the only argument is the size; the dtype/device become
    # keyword arguments so both the reference and the candidate share the exact
    # same call semantics.
    shape = plan.builder_args[0]
    return shape, {"dtype": dtype, "device": device}


def _build_inputs_fn_out(plan, dtype, device):
    # ``.out`` form: a pre-allocated buffer is passed as a keyword argument.
    shape = plan.builder_args[0]
    out = torch.empty(shape, dtype=dtype, device=device)
    return shape, {"out": out}


class EfficientZeroTensorBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark with allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=EFFICIENTZEROTENSOR_SHAPES)


# ``flag_gems._efficientzerotensor`` is not registered in every checkout;
# ``getattr(..., None)`` keeps the module importable while ``Benchmark._candidate_call``
# inside the benchmark still picks up the KernelGen override at runtime.
@pytest.mark._efficientzerotensor
def test__efficientzerotensor():
    bench = EfficientZeroTensorBenchmark(
        op_name="_efficientzerotensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._efficientzerotensor,
        gems_op=getattr(flag_gems, "_efficientzerotensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark._efficientzerotensor_out
def test__efficientzerotensor_out():
    bench = EfficientZeroTensorBenchmark(
        op_name="_efficientzerotensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._efficientzerotensor.out,
        gems_op=getattr(flag_gems, "_efficientzerotensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
