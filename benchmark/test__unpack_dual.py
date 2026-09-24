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

# ``_unpack_dual`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._unpack_dual`` and ``-m _unpack_dual``
# both work.
setattr(
    pytest.mark,
    "_unpack_dual",
    MarkDecorator(Mark("_unpack_dual", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor
# tangent) is the forward-mode AD dual-construction inverse: it reads the primal
# (as an aliasing view) and the tangent off a dual tensor created at an active
# forward-mode AD level. The named level must be live when the op is called, and
# forward-mode AD does not support nested dual_level() contexts, so the whole
# dual benchmark runs inside a single ``dual_level()`` and the level index it
# assigns is threaded through the input builder to both the reference
# (torch_op) and the candidate (gems_op). No public Benchmark family models a
# metadata/view op, so both overloads use a two-phase GenericBenchmark with
# case_fn + build_inputs_fn.
#
# gems_op is resolved through getattr because flag_gems._unpack_dual is not yet
# registered as a direct callable; KernelGen's --override _unpack_dual:<file>:<function> still wins at run time via Benchmark._candidate_call.

# A view op's latency is dominated by the call overhead, not by the tensor
# size. Capping the input numel avoids allocating multi-GB primal+tangent pairs
# (the generic DEFAULT_SHAPES include 1G-element tensors) for no signal.
MAX_NUMEL = 2**24  # 16M elements


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"level": "forward-ad"},
        builder_args=(shape,),
    )


def _plain_case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"level": "plain"},
        builder_args=(shape,),
    )


def _build_inputs_fn_factory(level):
    def _build_inputs_fn(plan, dtype, device):
        shape = plan.builder_args[0]
        primal = utils.generate_tensor_input(shape, dtype, device)
        tangent = utils.generate_tensor_input(shape, dtype, device)
        dual = torch.ops.aten._make_dual(primal, tangent, level)
        return dual, level

    return _build_inputs_fn


def _plain_build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, 0


class _UnpackDualBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over performance-relevant, capped shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [shape for shape in self.shapes if math.prod(shape) <= MAX_NUMEL]


@pytest.mark._unpack_dual
def test__unpack_dual():
    # The level must stay live for the entire benchmark run, so dual_level() is
    # entered here and the input builder closes over the exact level index it
    # assigns (a hardcoded level would break on torch builds whose forward AD
    # level counter is not zero-based).
    with torch.autograd.forward_ad.dual_level() as level:
        bench = _UnpackDualBenchmark(
            op_name="_unpack_dual",
            case_fn=_case_fn,
            build_inputs_fn=_build_inputs_fn_factory(level),
            torch_op=torch.ops.aten._unpack_dual,
            gems_op=getattr(flag_gems, "_unpack_dual", None),
            dtypes=consts.FLOAT_DTYPES,
        )
        bench.run()


@pytest.mark._unpack_dual
def test__unpack_dual_plain():
    # The tangent-None path: a plain tensor has no forward tangent at level 0,
    # so unpacking only produces the aliasing primal view. It is timed outside
    # any dual_level() context, matching how the op is called on a non-dual
    # tensor.
    bench = _UnpackDualBenchmark(
        op_name="_unpack_dual",
        case_fn=_plain_case_fn,
        build_inputs_fn=_plain_build_inputs_fn,
        torch_op=torch.ops.aten._unpack_dual,
        gems_op=getattr(flag_gems, "_unpack_dual", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
