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

"""Benchmark for ``aten::_convert_indices_from_coo_to_csr``.

Every workload is rank 1 -- a rank >= 2 index tensor is rejected natively -- so the cost
is driven by the index count (nnz) and by the row count, which is what the descriptors
below carry. The row count is not limited by the index dtype: a narrow index vector over
many rows is valid and the trailing rows simply stay empty, so a listed ``rows`` is
passed to both operators unchanged and only the generated indices are narrowed to what
the dtype can hold.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

OP_NAME = "_convert_indices_from_coo_to_csr"

# Workloads as ``(nnz, rows)`` descriptors: ``nnz`` is the length of the rank-1 sorted
# index vector that is actually built and executed, ``rows`` is exactly the ``size``
# handed to both operators. A ``--shape_file`` entry for this operator (or for this
# benchmark class) overrides them and must use the same pair format; the inherited
# generic 2-D/3-D shapes are not valid geometry for this operator.
COO_TO_CSR_CASES = [
    (0, 1),
    (1, 1),
    (65_536, 4_096),
    (1_048_576, 16_384),
    (16_777_216, 65_536),
    (67_108_864, 262_144),
    (268_435_456, 1_048_576),
    # Row-count boundaries: an empty table, an empty index vector over many rows, and a
    # dense table whose row count matches the index count.
    (0, 0),
    (0, 4_096),
    (256, 65_536),
    (16_384, 16_384),
]

# consts.INT_DTYPES / consts.EXTRA_INT_DTYPES are dtype pools, not capability gates.
# int64 input support is published as a device capability on the runtime device object;
# the same flag selects the listed and the executed dtypes.
BENCH_DTYPES = [
    dtype
    for dtype in list(consts.INT_DTYPES) + list(consts.EXTRA_INT_DTYPES)
    if dtype is not torch.int64 or flag_gems.runtime.device.support_int64
]


def _descriptor(entry, dtype):
    """Validate one ``(nnz, rows)`` descriptor and derive its index bound.

    ``rows`` is never reduced: the pointer table always has ``size + 1`` entries and the
    trailing unoccupied rows are valid, so a large ``size`` with narrow indices is a
    legitimate workload. Only the indices have to be storable in ``dtype``, so the
    exclusive index bound is ``min(rows, int(torch.iinfo(dtype).max) + 1)`` while
    ``rows`` is passed on as ``size`` unchanged. Returns ``(nnz, rows, index_upper)``.
    """
    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
        raise ValueError(
            f"{OP_NAME}: each shape entry must be a (nnz, rows) pair for the rank-1 "
            f"index vector that is built and executed; a rank-0/scalar or rank>=2 "
            f"entry such as {entry!r} is rejected rather than listed"
        )
    nnz, rows = entry
    for name, value in (("nnz", nnz), ("rows", rows)):
        # bool is an int subclass, so it is rejected before the integer check.
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{OP_NAME}: {name} must be a non-negative int, got {value!r}"
            )
    if dtype.is_floating_point or dtype == torch.bool:
        raise ValueError(f"{OP_NAME}: {dtype} inputs are not supported")
    if nnz and rows < 1:
        # A non-empty index vector needs at least one row: with ``size=0`` the result
        # buffer holds only ``out[0]`` and any index update would land outside it.
        raise ValueError(
            f"{OP_NAME}: rows must be at least 1 for a non-empty index vector"
        )
    index_upper = min(rows, int(torch.iinfo(dtype).max) + 1)
    if nnz and index_upper < 1:
        raise ValueError(
            f"{OP_NAME}: {dtype} cannot hold an index for {nnz} indices over "
            f"{rows} rows"
        )
    return nnz, rows, index_upper


def _case_fn(shape, dtype):
    nnz, rows, index_upper = _descriptor(shape, dtype)
    for out_int32 in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"input": (nnz,)},
            params={"size": rows, "out_int32": out_int32},
            builder_args=(nnz, index_upper),
        )


def _build_inputs_fn(plan, dtype, device):
    nnz, index_upper = plan.builder_args
    if nnz == 0:
        inp = torch.empty(0, dtype=dtype, device=device)
    else:
        # Sorted row indices in [0, index_upper), the operator input contract.
        # torch.randint validates ``high - 1`` against its accumulator dtype, so a row
        # table above INT32_MAX needs an int64 draw; the cast below stays exact because
        # ``_descriptor`` keeps index_upper inside what the input dtype can store.
        int32_max = torch.iinfo(torch.int32).max
        draw = torch.int32 if index_upper - 1 <= int32_max else torch.int64
        inp = torch.randint(0, index_upper, (nnz,), dtype=draw, device=device)
        inp = inp.sort().values.to(dtype)
    # The dict becomes call kwargs, so both operators run
    # op(input, size=rows, out_int32=out_int32).
    return inp, {"size": plan.params["size"], "out_int32": plan.params["out_int32"]}


class ConvertIndicesFromCooToCsrBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark using this operator's rank-1 descriptors."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=COO_TO_CSR_CASES)


@pytest.mark.convert_indices_from_coo_to_csr
def test__convert_indices_from_coo_to_csr():
    bench = ConvertIndicesFromCooToCsrBenchmark(
        op_name=OP_NAME,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._convert_indices_from_coo_to_csr,
        gems_op=getattr(flag_gems, OP_NAME, None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
