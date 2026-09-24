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

# ``_add_batch_dim`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._add_batch_dim`` and ``-m
# _add_batch_dim`` both work.
setattr(
    pytest.mark,
    "_add_batch_dim",
    MarkDecorator(Mark("_add_batch_dim", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_add_batch_dim is a zero-copy lazy wrapper: it hides one physical
# dimension of the input behind a vmap batch dimension and allocates nothing,
# so the benchmark measures dispatch and BatchedTensorImpl construction
# overhead. The default shape set contains a 1-B-element 1-D tensor whose cost
# would be dominated by input allocation; use allocation-friendly shapes whose
# ranks are >= 2 so a mid-range batch_dim (1 or 2) is always valid.
ADD_BATCH_DIM_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 256, 256),
    (20, 320, 15),
]


def _case_fn(shape, dtype):
    del dtype
    # batch_dim=1 is valid for every shape above (all ranks >= 2); level=0 is
    # the common non-nested vmap case.
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"batch_dim": 1, "level": 0},
        builder_args=(shape, 1, 0),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, batch_dim, level = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"batch_dim": batch_dim, "level": level}


class AddBatchDimBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=ADD_BATCH_DIM_SHAPES)


@pytest.mark._add_batch_dim
def test__add_batch_dim():
    bench = AddBatchDimBenchmark(
        op_name="_add_batch_dim",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._add_batch_dim,
        gems_op=getattr(flag_gems, "_add_batch_dim", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
