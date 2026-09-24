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

# ``_remove_batch_dim`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly on
# the MarkGenerator so ``@pytest.mark._remove_batch_dim`` and ``-m
# _remove_batch_dim`` both work.
setattr(
    pytest.mark,
    "_remove_batch_dim",
    MarkDecorator(Mark("_remove_batch_dim", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_remove_batch_dim inserts a broadcast batch dimension of size
# ``batch_size`` at position ``out_dim`` of the input shape (exactly
# ``self.expand(sizes)`` with ``batch_size`` inserted at ``out_dim``). It is a
# zero-copy view, so the benchmark measures dispatch and view-construction
# overhead rather than memory traffic. The default shape set contains a
# 1-B-element 1-D tensor whose cost would be dominated by input allocation, so
# the allocation-friendly shapes below are used instead. Each case is a
# (shape, out_dim, batch_size) triple chosen so the insert is a valid expand:
# either the batch dim is prepended with batch_size=1 (out_dim=0, purely a new
# leading dim), or it is inserted at out_dim=1 with batch_size equal to the
# leading input dim (the common vmap-unwrap pattern, which broadcasts along the
# new stride-0 batch dim).
_BENCH_CASES = [
    ((256,), 1, 256),
    ((64, 64), 1, 64),
    ((1024, 1024), 0, 1),
    ((1024, 1024), 1, 1024),
    ((4096, 4096), 0, 1),
    ((64, 512, 512), 1, 64),
    ((128, 256, 256), 1, 128),
    ((20, 320, 15), 1, 20),
    ((16, 7, 57, 32, 29), 0, 1),
]


def _case_fn(case, dtype):
    del dtype
    shape, out_dim, batch_size = case
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"level": 0, "batch_size": batch_size, "out_dim": out_dim},
        builder_args=case,
    )


def _build_inputs_fn(plan, dtype, device):
    shape, out_dim, batch_size = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"level": 0, "batch_size": batch_size, "out_dim": out_dim}


class RemoveBatchDimBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark._remove_batch_dim
def test__remove_batch_dim():
    bench = RemoveBatchDimBenchmark(
        op_name="_remove_batch_dim",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._remove_batch_dim,
        gems_op=getattr(flag_gems, "_remove_batch_dim", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
