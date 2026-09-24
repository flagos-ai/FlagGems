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

# ``_neg_view`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._neg_view`` and ``-m _neg_view`` both
# work.
setattr(
    pytest.mark,
    "_neg_view",
    MarkDecorator(Mark("_neg_view", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_neg_view is a zero-copy negative view: it shares the input storage and
# only toggles the lazy neg bit, so the benchmark measures dispatch and
# view-construction overhead. No public Benchmark family models a view op, so
# this uses the two-phase GenericBenchmark (case_fn + build_inputs_fn). The
# default shape set is dominated by 1G/268M-element inputs whose allocation
# cost swamps the signal; use allocation-friendly shapes instead.
NEG_VIEW_SHAPES = [
    (2, 2),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4, 8, 16, 32),
    (64, 128, 256),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {}


class NegViewBenchmark(OperatorBenchmark):
    # A view op's latency is dominated by the call overhead rather than the
    # tensor size, so cap the input numel to avoid allocating multi-GB inputs
    # for no signal.
    MAX_NUMEL = 2**24  # 16M elements

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=NEG_VIEW_SHAPES)
        self.shapes = [s for s in self.shapes if math.prod(s) <= self.MAX_NUMEL]


@pytest.mark._neg_view
def test__neg_view():
    bench = NegViewBenchmark(
        op_name="_neg_view",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._neg_view,
        # flag_gems._neg_view is not registered as a direct callable yet;
        # KernelGen's --override _neg_view:<file>:<function> still wins at run time
        # through Benchmark._candidate_call.
        gems_op=getattr(flag_gems, "_neg_view", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
