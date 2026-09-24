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
from . import base, consts, utils

# ``_has_same_storage_numel`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._has_same_storage_numel`` and
# ``-m _has_same_storage_numel`` both work.
setattr(
    pytest.mark,
    "_has_same_storage_numel",
    MarkDecorator(
        Mark("_has_same_storage_numel", (), {}, _ispytest=True), _ispytest=True
    ),
)

# aten::_has_same_storage_numel(Tensor self, Tensor other) -> bool is a pure
# storage-metadata query: it compares the two storage element counts and reads
# no payload, so the benchmark mostly measures dispatch overhead. No public
# Benchmark family covers a two-tensor predicate returning a bool, hence the
# two-phase GenericBenchmark below (case_fn + build_inputs_fn).
#
# The default core-shape set contains a 1-B-element 1-D tensor whose cost would
# be dominated by input allocation; use allocation-friendly shapes instead so
# the measured latency reflects the operator itself.
_HAS_SAME_STORAGE_NUMEL_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 256, 256),
    (20, 320, 15),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"self": shape, "other": shape},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    self_inp = utils.generate_tensor_input(shape, dtype, device)
    other_inp = utils.generate_tensor_input(shape, dtype, device)
    return self_inp, other_inp


class _HasSameStorageNumelBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(
            shape_file_path, default_shapes=_HAS_SAME_STORAGE_NUMEL_SHAPES
        )


@pytest.mark._has_same_storage_numel
def test__has_same_storage_numel():
    bench = _HasSameStorageNumelBenchmark(
        op_name="_has_same_storage_numel",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._has_same_storage_numel,
        gems_op=getattr(flag_gems, "_has_same_storage_numel", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
