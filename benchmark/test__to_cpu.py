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

"""Benchmark for ``aten::_to_cpu``.

``_to_cpu`` copies a ``Tensor[]`` to the host, so the measured cost is
host/device traffic. Shapes come from the framework shape file: ``_to_cpu`` has
no entry of its own, so the base class resolves the shared ``Benchmark`` entry
and merges the class's own comprehensive extras on top; an explicit
``--shape_file`` still replaces them. Listing stays tensor-free because
``case_fn`` only plans metadata. ``torch_op`` is the ATen reference (perf
baseline) and ``gems_op`` the FlagGems candidate; both are called as
``op(tensors)``.
"""

import pytest
import torch

import flag_gems

from . import base, consts

_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name))


# The operator is a dtype-agnostic storage relocation, so every supported dtype
# family is timed: float, complex, int and bool.
BENCH_DTYPES = [
    dtype
    for dtype in (
        consts.FLOAT_DTYPES
        + consts.COMPLEX_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + [torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2]
    )
    if _dtype_supported(dtype)
]


def _case_fn(shape, dtype):
    del dtype
    # The aten signature is a Tensor[], so every case plans a two-tensor list and
    # the list length, not a single-element call, is what gets measured.
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"num_tensors": 2},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    tensors = [
        torch.empty(shape, dtype=dtype, device=device)
        for _ in range(plan.params["num_tensors"])
    ]
    # Flat positional arguments with a trailing kwargs dict: the tensor list is
    # passed as one positional argument, matching ``op(tensors)``.
    return tensors, {}


@pytest.mark.to_cpu
def test__to_cpu():
    bench = base.GenericBenchmark(
        op_name="_to_cpu",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._to_cpu,
        gems_op=getattr(flag_gems, "_to_cpu", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
