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

import math

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# ``_make_dual`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._make_dual`` and ``-m _make_dual`` both
# work.
setattr(
    pytest.mark,
    "_make_dual",
    MarkDecorator(Mark("_make_dual", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)
# attaches a forward tangent to an aliasing view of the primal at an active
# forward-mode AD level. The level the op names must be live when it is called,
# and forward-mode AD does not support nested dual_level() contexts, so the
# whole benchmark runs inside a single ``dual_level()`` and the level index it
# assigns is threaded through the input builder to both the reference
# (torch_op) and the candidate (gems_op). The candidate kernel only consumes
# primal's data (the tangent is dual metadata), so each case materializes
# primal + tangent and reads primal once. No public Benchmark family models a
# metadata/view op, so this uses a two-phase GenericBenchmark with case_fn +
# build_inputs_fn.
#
# gems_op is resolved through getattr because flag_gems._make_dual is not yet
# registered as a direct callable; KernelGen's --override _make_dual:<file>:<function> still wins at run time via Benchmark._candidate_call.

# A view op's latency is dominated by the call overhead, not by the tensor
# size. Capping the input numel avoids allocating multi-GB primal+tangent pairs
# (the generic DEFAULT_SHAPES include 1G-element tensors) for no signal.
MAX_NUMEL = 2**24  # 16M elements


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"primal": shape, "tangent": shape},
        params={"level": "forward-ad-active"},
        builder_args=(shape,),
    )


class MakeDualBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over performance-relevant, capped shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [shape for shape in self.shapes if math.prod(shape) <= MAX_NUMEL]


@pytest.mark._make_dual
def test__make_dual():
    # The level must stay live for the entire benchmark run, so dual_level() is
    # entered here and the input builder closes over the exact level index it
    # assigns (a hardcoded level would break on torch builds whose forward AD
    # level counter is not zero-based).
    with torch.autograd.forward_ad.dual_level() as level:

        def _build_inputs_fn(plan, dtype, device):
            shape = plan.builder_args[0]
            primal = utils.generate_tensor_input(shape, dtype, device)
            tangent = utils.generate_tensor_input(shape, dtype, device)
            return primal, tangent, level

        bench = MakeDualBenchmark(
            op_name="_make_dual",
            case_fn=_case_fn,
            build_inputs_fn=_build_inputs_fn,
            torch_op=torch.ops.aten._make_dual,
            gems_op=getattr(flag_gems, "_make_dual", None),
            dtypes=consts.FLOAT_DTYPES,
        )
        bench.run()
