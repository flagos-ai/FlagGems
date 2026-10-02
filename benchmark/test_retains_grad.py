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

"""Benchmark for ``aten::retains_grad``.

The operator reports Autograd state and returns a Python ``bool``, so
``torch_op`` (the perf comparison reference) and ``gems_op`` (the candidate)
are both called as ``op(input)``. No public Benchmark family covers a state
query, so this file uses the two-phase ``case_fn``/``build_inputs_fn`` API of
``GenericBenchmark``: one plan set feeds ``--list-cases`` and normal execution,
and the operator keeps the shared default/comprehensive shape grid and
``--shape_file`` support because its schema constrains neither rank nor extent.
"""

import pytest
import torch

import flag_gems

from . import base, consts

# Both answers are planned: a leaf answers False and a retained non-leaf answers
# True. The state is metadata, so --list-cases stays free of tensor allocation.
_GRAD_STATES = ("grad_leaf", "nonleaf", "retained_nonleaf")

# The benchmarked dtypes follow the correctness coverage groups -- the nine
# required dtypes plus bool, complex64 and float64 -- and drop only the types
# the backend does not provide (static capability flags, no probing).
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


BENCH_DTYPES = [
    dtype
    for dtype in list(consts.FLOAT_DTYPES)
    + [
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.complex64,
        torch.float64,
    ]
    if _dtype_supported(dtype)
]


def _case_fn(shape, dtype):
    states = ("leaf",) + (
        _GRAD_STATES if dtype.is_floating_point or dtype.is_complex else ()
    )
    for state in states:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"state": state},
            builder_args=(shape, state),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, state = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    if state != "leaf":
        inp.requires_grad_(True)
    if state in ("nonleaf", "retained_nonleaf"):
        inp = inp.view(shape)
    if state == "retained_nonleaf":
        inp.retain_grad()
    return inp, {}


@pytest.mark.retains_grad
def test_retains_grad():
    bench = base.GenericBenchmark(
        op_name="retains_grad",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.retains_grad,
        gems_op=getattr(flag_gems, "retains_grad", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
