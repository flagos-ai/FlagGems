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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark
from .test_aminmax import _AminmaxStaticReference


class _AminmaxDirectBenchmark(_AminmaxStaticReference, OperatorBenchmark):
    _H800_PROFILE = "direct"


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    yield base.BenchmarkCasePlan(shape={"self": list(shape)}, builder_args=(shape,))


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    return (utils.generate_tensor_input(shape, dtype, device),)


@pytest.mark.underscore_aminmax
@pytest.mark.skip_native(
    vendors=["enflame"],
    reason="native private _aminmax is CPU fallback; require exact direct H800 profile",
)
def test__aminmax():
    _AminmaxDirectBenchmark(
        op_name="_aminmax",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch._aminmax,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
