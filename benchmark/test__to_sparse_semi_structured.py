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

"""Benchmark tests for ``aten::_to_sparse_semi_structured``."""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

_OP_NAME = "_to_sparse_semi_structured"

# Performance-relevant descriptors, all aligned for every benchmark dtype, so no
# requested workload is rounded up, replaced or silently dropped.
_BENCH_CASES = (
    (64, 64),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
)

# Measured per-dtype geometry. A requested descriptor is validated against its own
# dtype and rejected - never rewritten upward - when it is not representable, because
# the sparse slots are compressed in units of these alignment values.
_MIN_GEOMETRY = {
    torch.float16: (32, 32),
    torch.bfloat16: (32, 32),
    torch.int8: (16, 64),
    torch.float32: (32, 16),
}

_BENCH_DTYPES = [torch.float16, torch.float32, torch.int8]
if flag_gems.runtime.device.support_bf16:
    _BENCH_DTYPES.insert(1, torch.bfloat16)


def _check_descriptor(shape, dtype):
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("descriptor {0!r} is not a (rows, columns) pair".format(shape))
    rows, cols = shape
    if any(isinstance(extent, bool) for extent in (rows, cols)) or any(
        not isinstance(extent, int) or extent < 0 for extent in (rows, cols)
    ):
        raise ValueError(
            "descriptor {0!r} is not a pair of non-negative integers".format(shape)
        )
    min_rows, min_cols = _MIN_GEOMETRY[dtype]
    if rows % min_rows or cols % min_cols:
        raise ValueError(
            "descriptor {0!r} is not representable for {1}: this dtype requires "
            "rows % {2} and columns % {3}".format(shape, dtype, min_rows, min_cols)
        )


def _structured_dense(shape, dtype, device):
    # Dense input holding exactly the retained slots of the dtype's sparse layout, so
    # the measured kernel does the packing work instead of unpacking a full matrix.
    rows, cols = shape
    group = 2 if dtype == torch.float32 else 4
    col = torch.arange(cols, device=device, dtype=torch.int32).view(1, cols)
    slot = col % group
    keep = (slot == 0) if group == 2 else ((slot == 0) | (slot == 1))
    values = ((((col * 7 + 1) % 61) + 1).to(dtype)).expand(rows, cols)
    return torch.where(keep, values, torch.zeros((), dtype=dtype, device=device))


def _case_fn(shape, dtype):
    # Planning only: the descriptor is validated here, before any allocation, so
    # listing and execution walk exactly the same cases.
    _check_descriptor(shape, dtype)
    yield base.BenchmarkCasePlan(
        shape={"input": tuple(shape)},
        params={"rows": shape[0], "columns": shape[1]},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return _structured_dense(shape, dtype, device), {}


class StructuredSparseBenchmark(OperatorBenchmark):
    """Two-phase benchmark over the operator's own descriptors.

    ``OperatorBenchmark`` resolves an explicit shape file by operator key first and
    by this class name second, raises for a missing or malformed file, and keeps an
    explicit empty ``shapes`` override empty instead of restoring the defaults.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark.to_sparse_semi_structured
@pytest.mark.parametrize("dtype", _BENCH_DTYPES)
def test__to_sparse_semi_structured(dtype):
    StructuredSparseBenchmark(
        op_name=_OP_NAME,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._to_sparse_semi_structured,
        gems_op=getattr(flag_gems, _OP_NAME, None),
        dtypes=[dtype],
    ).run()
