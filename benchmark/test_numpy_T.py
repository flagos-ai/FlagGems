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

# aten::numpy_T only rearranges metadata, so the benchmark times dispatch plus
# view construction. It writes nothing, so fresh_inputs/is_inplace do not apply.
# No rank is filtered: 0-D and 1-D are valid native inputs too, and every rank up
# to the framework maximum is accepted, so the shared default and COMPREHENSIVE
# shape sets (and any --shape_file override) stay in effect.
_DEVICE = flag_gems.runtime.device

# The op is storage-agnostic; only the optional storage types are gated, and no
# dtype is dropped in quick mode.
_BENCH_DTYPES = (
    [torch.float16, torch.float32]
    + ([torch.bfloat16] if getattr(_DEVICE, "support_bf16", False) else [])
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + (
        [torch.float8_e4m3fn, torch.float8_e5m2]
        if getattr(_DEVICE, "support_fp8", False)
        else []
    )
    + consts.COMPLEX_DTYPES
    + consts.BOOL_DTYPES
)


def _case_fn(shape, dtype):
    del dtype  # numpy_T has no dtype-dependent parameter
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    # The op never reads element values, so the input only needs its shape and
    # strides; uninitialized storage avoids paying a fill cost per case.
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, {}


@pytest.mark.numpy_T
def test_numpy_T():
    bench = base.GenericBenchmark(
        op_name="numpy_T",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.numpy_T,
        gems_op=getattr(flag_gems, "numpy_T", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
