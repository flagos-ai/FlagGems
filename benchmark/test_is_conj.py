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

import flag_gems

from . import base, consts, utils

# aten::is_conj(Tensor self) -> bool only reads the lazy conjugate bit and runs no
# kernel. The plans below vary that bit (plain / conjugate / resolved conjugate /
# conjugate + negative) instead of the payload, and the state is applied in the
# builder only, so listing stays tensor-free. No public Benchmark family covers a
# bool-returning metadata predicate, hence the two-phase GenericBenchmark, which
# keeps the framework shape set and its --shape-file support.
_STATES = ("plain", "conj", "conj_resolved", "neg_of_conj")


def _case_fn(shape, dtype):
    del dtype
    for state in _STATES:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"state": state},
            builder_args=(shape, state),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, state = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if state == "conj":
        inp = inp.conj()
    elif state == "conj_resolved":
        inp = inp.conj().resolve_conj()
    elif state == "neg_of_conj":
        inp = torch._neg_view(inp.conj())
    return inp, {}


@pytest.mark.is_conj
def test_is_conj():
    bench = base.GenericBenchmark(
        op_name="is_conj",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_conj,
        gems_op=getattr(flag_gems, "is_conj", None),
        dtypes=consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.run()
