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

# ``_fw_primal`` starts with an underscore and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names, so register it on the
# MarkGenerator directly (``@pytest.mark._fw_primal`` and ``-m _fw_primal``).
setattr(
    pytest.mark,
    "_fw_primal",
    MarkDecorator(Mark("_fw_primal", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_fw_primal(Tensor(a) self, int level) -> Tensor(a) is a zero-copy
# forward-mode-AD view: it shares the input storage and allocates nothing, so
# the benchmark measures dispatch and view-construction overhead rather than
# memory traffic. The default shape set is dominated by 1G/268M-element inputs
# whose allocation cost swamps the signal, so use allocation-friendly shapes
# instead (the largest case below is a 128 MiB fp32 tensor).
FW_PRIMAL_SHAPES = [
    (1024,),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (64, 128, 128),
    (8, 256, 512),
]


def _case_fn(shape, dtype):
    del dtype
    # The op needs a ``level`` alongside the tensor, which the public unary
    # pointwise families do not supply, so the two-phase GenericBenchmark
    # forwards the constant level explicitly (documented level 0).
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"level": 0},
        builder_args=(shape, 0),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, level = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {"level": level}


class FwPrimalBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to allocation-friendly shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=FW_PRIMAL_SHAPES)


@pytest.mark._fw_primal
def test__fw_primal():
    bench = FwPrimalBenchmark(
        op_name="_fw_primal",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._fw_primal,
        # flag_gems._fw_primal is not registered as a direct callable yet;
        # KernelGen's --override _fw_primal:<file>:<function> still wins at run time
        # through Benchmark._candidate_call.
        gems_op=getattr(flag_gems, "_fw_primal", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
