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

import math

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# A metadata query reads the bit recorded on a sparse COO tensor, so a workload
# is (logical size, nnz, recorded bit). Explicit descriptors join the shared
# shape grid read from the shape file; a plain shape from that grid expands into
# the same descriptor with a deterministic nnz = numel // 8, which satisfies the
# only schema constraint (nnz <= numel) at every shared size.
IS_COALESCED_CASES = [
    ((64,), 32, False),
    ((1024, 1024), 65536, False),
    ((1024, 1024), 65536, True),
    ((4096, 4096), 262144, False),
    ((4096, 4096), 262144, True),
    ((256, 256, 256), 1048576, False),
    ((20, 320, 15), 262144, False),
    ((16, 7, 57, 32, 29), 262144, True),
]

_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.int64: "support_int64",
    torch.bfloat16: "support_bf16",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


# The nine required benchmark dtypes plus float64 / int16 / bool: the stored
# values never influence a metadata query, but every dtype the operator accepts
# stays represented.
_REQUIRED_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
]

BENCH_DTYPES = [
    dtype
    for dtype in _REQUIRED_DTYPES
    + [torch.float64, torch.int16, torch.bool, torch.complex64]
    if _dtype_supported(dtype)
]


def _is_descriptor(spec):
    return len(spec) == 3 and isinstance(spec[0], (tuple, list))


def _flat_positions(size, nnz, *, ordered, device):
    """Deterministic coordinates; see the correctness test for the contract."""
    if nnz == 0:
        return torch.empty((len(size), 0), dtype=torch.int64, device=device)
    if not size:
        return torch.empty((0, nnz), dtype=torch.int64, device=device)
    numel = math.prod(size)
    if ordered:
        flat = torch.arange(nnz, dtype=torch.int64) * numel // nnz
    else:
        flat = torch.arange(nnz, dtype=torch.int64).flip(0) % numel
    indices = torch.empty((len(size), nnz), dtype=torch.int64)
    for dim in reversed(range(len(size))):
        indices[dim] = flat % size[dim]
        flat = flat // size[dim]
    return indices.to(device)


def _case_fn(shape, dtype):
    del dtype
    if _is_descriptor(shape):
        size, nnz, recorded = shape
        yield base.BenchmarkCasePlan(
            shape={"size": list(size)},
            params={"nnz": int(nnz), "recorded_bit": bool(recorded)},
            builder_args=(tuple(size), int(nnz), bool(recorded)),
        )
        return
    size = tuple(shape)
    numel = math.prod(size) if size else 1
    for recorded in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"size": list(size)},
            params={"nnz": numel // 8, "recorded_bit": recorded},
            builder_args=(size, numel // 8, recorded),
        )


def _build_inputs_fn(plan, dtype, device):
    size, nnz, recorded = plan.builder_args
    indices = _flat_positions(size, nnz, ordered=recorded, device=device)
    # Values are never read by the query; a deterministic fill avoids a random
    # payload whose generation would dominate a metadata-only benchmark.
    values = torch.zeros(nnz, dtype=dtype, device=device)
    inp = torch.sparse_coo_tensor(
        indices, values, tuple(size), device=device, is_coalesced=recorded
    )
    return inp, {}


class IsCoalescedBenchmark(OperatorBenchmark):
    """Keeps the shared shape grid (shape-file aware) and adds the sparse
    (size, nnz, recorded bit) descriptors that this operator actually reads."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(self.shapes) + IS_COALESCED_CASES


@pytest.mark.is_coalesced
def test_is_coalesced():
    bench = IsCoalescedBenchmark(
        op_name="is_coalesced",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_coalesced,
        gems_op=getattr(flag_gems, "is_coalesced", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
