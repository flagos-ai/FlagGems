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

"""Benchmark for aten::result_type.

result_type is pure dtype inference: its four overloads (.Tensor, .Scalar,
.Scalar_Tensor, .Scalar_Scalar) return a Python int ScalarType code, so what is
measured is dispatch plus CPU-side promotion, and no element value is ever read.
Payloads are therefore uninitialized torch.empty buffers of the dtype bucket and
are never compared. Operand shape still matters through the rank role: for two
different dtypes a 0-dim tensor is promoted like a Python scalar while the
dimensioned operand keeps priority, so the case list covers that mixed-dtype
rank role beside the four call forms. The tensor cases keep the library default,
comprehensive and caller shape grids unchanged; only Scalar_Scalar, which has no
tensor operand at all, is emitted once per dtype bucket instead of repeating an
identical case for every shape.
"""

import pytest
import torch

import flag_gems

from . import base, consts

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

_FLOAT_SCALAR = 1.5
_BOOL_SCALAR = True

# The four runtime overloads, all reached through the single public entry point
# flag_gems.result_type. tensor_scalar and scalar_tensor share one case list
# because a tensor and a Python scalar promote independently of the tensor shape
# (probed on the active backend), and mixed_rank is the tensor/tensor form whose
# second operand is a 0-dim tensor of a different dtype.
_TENSOR_FORMS = ("tensor_tensor", "mixed_rank", "tensor_scalar", "scalar_tensor")

# Float8 promotes only with itself (a mixed fp8 pair raises in the native op),
# so an fp8 bucket pairs its 0-dim operand with the same fp8 dtype.
_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]

# Every dtype whose storage the op accepts, taken from the shared lists; fp8 is
# included only when the running build exposes a usable fp8 storage type.
_BENCH_DTYPES = list(
    dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [torch.float64, torch.complex128]
        + _FP8_DTYPES
    )
)

_BENCH_DTYPES = [dtype for dtype in _BENCH_DTYPES if _DTYPE_FLAGS.get(dtype, True)]


def _rank_partner(dtype):
    """Dtype of the 0-dim companion operand for the mixed-dtype rank role."""
    if dtype in _FP8_DTYPES:
        return dtype
    return torch.float16 if dtype is torch.float32 else torch.float32


def _case_fn(shape, dtype):
    for form in _TENSOR_FORMS:
        partner = _rank_partner(dtype) if form == "mixed_rank" else dtype
        yield base.BenchmarkCasePlan(
            shape=(
                {"input": shape, "other": ()}
                if form == "mixed_rank"
                else {"input": shape}
            ),
            params={"form": form, "partner_dtype": str(partner)},
            builder_args=(form, shape, partner),
        )


def _scalar_scalar_plan():
    return base.BenchmarkCasePlan(
        shape={},
        params={"form": "scalar_scalar", "partner_dtype": ""},
        builder_args=("scalar_scalar", (), None),
    )


def _build_inputs_fn(plan, dtype, device):
    form, shape, partner = plan.builder_args
    if form == "tensor_tensor":
        return (
            torch.empty(shape, dtype=dtype, device=device),
            torch.empty(shape, dtype=dtype, device=device),
            {},
        )
    if form == "mixed_rank":
        return (
            torch.empty(shape, dtype=dtype, device=device),
            torch.empty((), dtype=partner, device=device),
            {},
        )
    if form == "tensor_scalar":
        return torch.empty(shape, dtype=dtype, device=device), _FLOAT_SCALAR, {}
    if form == "scalar_tensor":
        return _FLOAT_SCALAR, torch.empty(shape, dtype=dtype, device=device), {}
    return _FLOAT_SCALAR, _BOOL_SCALAR, {}


class ResultTypeBenchmark(base.GenericBenchmark):
    """Two-phase benchmark that keeps the shared tensor shape grids."""

    def get_case_iter(self, dtype):
        cases = list(super().get_case_iter(dtype))
        yield from cases
        # Scalar_Scalar carries no tensor, so one case per dtype bucket stands
        # in for all shapes instead of a redundant repeat for each of them.
        yield self._case_from_plan(dtype, len(cases), _scalar_scalar_plan())


@pytest.mark.result_type
def test_result_type():
    bench = ResultTypeBenchmark(
        op_name="result_type",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.result_type,
        gems_op=getattr(flag_gems, "result_type", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
