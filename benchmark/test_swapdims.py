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

# swapdims only permutes size/stride metadata, so the inputs are never read:
# torch.empty keeps fixtures cheap and every storable dtype stays a valid operand.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}

# Storable dtypes are all valid operands for a metadata swap; the capability map
# drops only those the runtime reports as unsupported.
_BENCH_DTYPE_CANDIDATES = (
    consts.FLOAT_DTYPES
    + [torch.float64]
    + consts.EXTRA_INT_DTYPES
    + consts.INT_DTYPES
    + consts.BOOL_DTYPES
    + consts.COMPLEX_DTYPES
    + [torch.complex128]
    + [dtype for dtype in consts.FP8_DTYPES if dtype is not None]
)


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


BENCH_DTYPES = [dtype for dtype in _BENCH_DTYPE_CANDIDATES if _dtype_supported(dtype)]

# Ranks the shared dense grids do not contain: 0-d identity, size-1 and empty.
BOUNDARY_SHAPES = [(), (1,), (1, 1), (0, 3), (5, 0, 7)]


def _dim_pairs(shape):
    rank = len(shape)
    if rank < 2:
        # A 0-d or 1-d operand accepts only the identity pair.
        return [(0, 0)]
    if rank == 2:
        return [(0, 1)]
    return [(0, 1), (0, rank - 1), (-1, -2)]


def _case_fn(shape, dtype):
    del dtype
    for dim0, dim1 in _dim_pairs(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"dim0": dim0, "dim1": dim1},
            builder_args=(shape, dim0, dim1),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, dim0, dim1 = plan.builder_args
    return torch.empty(shape, dtype=dtype, device=device), dim0, dim1


class SwapdimsBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Comprehensive level only: keep the shared extra grid and add the
        # boundary ranks this metadata swap should also be measured on.
        return list(dict.fromkeys(super().set_more_shapes() + BOUNDARY_SHAPES))


@pytest.mark.swapdims
def test_swapdims():
    bench = SwapdimsBenchmark(
        op_name="swapdims",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.swapdims,
        gems_op=getattr(flag_gems, "swapdims", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
