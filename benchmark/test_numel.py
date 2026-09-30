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

from . import base, consts

# aten::numel(Tensor self) -> int returns a host-side int and never reads the
# payload, so the two-phase GenericBenchmark is used (no public family covers a
# one-tensor metadata query) and the inputs stay uninitialized: filling them
# would add allocation cost to a shape-only query. The framework shape file and
# DEFAULT_SHAPES are kept, so a caller-supplied shape file still applies.


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(shape={"input": shape}, builder_args=(shape,))


def _build_inputs_fn(plan, dtype, device):
    inp = torch.empty(plan.builder_args[0], dtype=dtype, device=device)
    return inp, {}


@pytest.mark.numel
def test_numel():
    bench = base.GenericBenchmark(
        op_name="numel",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.numel,
        gems_op=getattr(flag_gems, "numel", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
