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

from . import base

# aten::q_zero_point only dereferences the quantizer stored on the tensor, so the
# measured call does not depend on the tensor contents and the shared default
# shapes (and any caller-supplied shape file) are used unchanged. The storage
# dtype is private builder state; the quantized dtype is the reported case dtype.
_QUANT_TO_STORAGE = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
}
_QUANT_DTYPES = list(_QUANT_TO_STORAGE)
_SCALE = 0.5
_ZERO_POINT = 7


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"q_dtype": str(dtype), "scale": _SCALE, "zero_point": _ZERO_POINT},
        builder_args=(shape, _QUANT_TO_STORAGE[dtype]),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype  # the quantized input dtype lives in builder_args
    shape, storage_dtype = plan.builder_args
    storage = torch.zeros(shape, dtype=storage_dtype, device=device)
    inp = torch.ops.aten._make_per_tensor_quantized_tensor(storage, _SCALE, _ZERO_POINT)
    return (inp,)


@pytest.mark.q_zero_point
def test_q_zero_point():
    bench = base.GenericBenchmark(
        op_name="q_zero_point",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.q_zero_point,
        gems_op=getattr(flag_gems, "q_zero_point", None),
        dtypes=_QUANT_DTYPES,
    )
    bench.run()
