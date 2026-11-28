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


@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
@pytest.mark.max
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_max():
    bench = base.UnaryReductionBenchmark(
        op_name="max", torch_op=torch.max, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
@pytest.mark.max_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_max_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="max_dim", torch_op=torch.max, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def _max_backward_values(inp, dim=None):
    """Native reference of the timed autograd chain for both torch.max overloads.

    UnaryReductionBenchmark yields a 1-D tensor with no dim (full reduction via
    `torch.max(x)`) and N-D tensors with dim=1 (`torch.max(x, dim).values`). With
    `is_backward=True` the harness times `torch.autograd.grad` on the result, so
    only the values output is returned here.
    """
    if dim is None:
        return torch.max(inp)
    return torch.max(inp, dim=dim).values


def _max_backward_gems(inp, dim=None):
    """Same chain through the FlagGems entries, including their backward."""
    if dim is None:
        return flag_gems.max(inp)
    return flag_gems.max_dim(inp, dim=dim).values


@pytest.mark.max_backward
@pytest.mark.max
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_max_backward():
    """Autograd latency of the max backward.

    ATen exposes no `max_backward` operator to intercept, so the candidate is
    the FlagGems public entry (`flag_gems.max` / `flag_gems.max_dim`) whose
    `torch.autograd.Function` backward launches the Triton kernels. The baseline
    is the native `torch.max` autograd formula on the same inputs.
    """
    dtypes = consts.FLOAT_DTYPES
    if flag_gems.vendor_name == "enflame":
        # The baseline leg is the native max.dim autograd chain, whose int64
        # index output topsaten only supports for int64/fp32/int32 data. For the
        # 16-bit dtypes the native call aborts the process, so only the fp32
        # chain can be timed on this backend.
        dtypes = [dtype for dtype in dtypes if dtype == torch.float32]
    bench = base.UnaryReductionBenchmark(
        op_name="max_backward",
        torch_op=_max_backward_values,
        gems_op=_max_backward_gems,
        dtypes=dtypes,
        is_backward=True,
    )
    bench.run()
