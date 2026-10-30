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


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"mode": 3, "value": 1.5},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, device=device, dtype=dtype)
    rank = inp.ndim
    # mode 3 == constant (see aten padding_mode enum: 0 reflect, 1 replicate,
    # 2 circular, 3 constant). Constant padding works for any rank/shape, so it
    # is the representative case for the benchmark sweep.
    pad = [3 for _ in range(rank * 2)]
    return inp, pad, 3, 1.5


@pytest.mark.underscore_pad_enum
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_pad_enum():
    bench = base.GenericBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="_pad_enum",
        torch_op=torch.ops.aten._pad_enum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
