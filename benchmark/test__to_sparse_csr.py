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
from .generated_operator_utils import OperatorBenchmark

# Core dense workloads as (shape, dense_dim, call form). The call form is part of
# the descriptor, so 'omitted' calls the operator without dense_dim, 'none' passes
# None explicitly and 'int' passes the value; the same descriptor grammar is read
# from a user shape file, where a row may be just the shape (the dense_dim is then
# left out of the call) or (shape, dense_dim[, call form]). Rows below rank - 2
# (batched/hybrid) need an equal number of specified elements in every batch, and
# every row must keep a non-zero product over the batch axes.
_BENCH_CASES = [
    ((1024, 1024), None, "omitted"),
    ((2048, 2048), 0, "int"),
    ((20, 320, 15), 1, "int"),
    ((16, 128, 64, 60), 2, "int"),
    ((16, 7, 57, 32, 29), 3, "int"),
    ((16, 8, 16, 32), 1, "int"),
    ((4, 8, 16), 0, "int"),
]

# bfloat16 is the only optional entry of consts.FLOAT_DTYPES on the active
# backend; the static capability flag decides it without a device probe.
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]

_CALL_FORMS = ("omitted", "none", "int")

# Textual spellings of a missing value that a shape file may carry.
_NULL_TOKENS = frozenset({"none", "null", "~", ""})


def _row_counts(rows, cols):
    """Specified elements per row of the patterned operand.

    Empty rows at the leading, middle and trailing position, one single element
    row, one two element row and, for narrow matrices, one full row, so sparse
    packing is exercised instead of a fully dense operand.
    """
    counts = [0] * max(rows, 0)
    if rows <= 0 or cols <= 0:
        return counts
    for row in range(rows):
        if rows >= 4 and (row == 0 or row == rows - 1 or row == rows // 2):
            continue
        if row % 4 == 3 and cols <= 64:
            counts[row] = cols
        elif row % 4 == 1:
            counts[row] = 1
        else:
            counts[row] = min(2, cols)
    return counts


def _descriptor_shape(spec):
    """Validate the shape field of one descriptor."""
    if not isinstance(spec, (tuple, list)):
        raise ValueError(
            f"_to_sparse_csr benchmark shape {spec!r}: expected a list of extents"
        )
    spec = tuple(spec)
    for size in spec:
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"_to_sparse_csr benchmark shape {spec!r}: expected integer extents >= 0"
            )
    if len(spec) < 2:
        raise ValueError(
            f"_to_sparse_csr benchmark shape {spec!r}: the operator requires rank >= 2"
        )
    return spec


def _descriptor_dense_dim(value):
    """Normalise the dense_dim field of one descriptor.

    A shape file stores scalars only, so an absent dense_dim arrives either as a
    real null or as one of its textual spellings; both mean "no value given".
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in _NULL_TOKENS:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"_to_sparse_csr benchmark dense_dim {value!r}: expected an integer or null"
        )
    return value


def _case_descriptor(row):
    """Validate one benchmark row: a shape, or (shape, dense_dim[, call form]).

    Listing and execution share these descriptors, so a malformed entry - in the
    built-in list or in a user shape file - fails here with the offending value
    instead of being silently reinterpreted as a different workload. The call form
    defaults to 'int' when a dense_dim is given and to 'omitted' otherwise.
    """
    if not isinstance(row, (tuple, list)) or not row:
        raise ValueError(f"_to_sparse_csr benchmark case {row!r}: expected a list")
    if all(not isinstance(field, (tuple, list)) for field in row):
        # A bare shape listed directly in a shape file, e.g. [1024, 1024].
        row = (row,)
    if len(row) > 3:
        raise ValueError(
            f"_to_sparse_csr benchmark case {row!r}: expected at most "
            "(shape, dense_dim, call form)"
        )
    spec = _descriptor_shape(row[0])
    dense_dim = _descriptor_dense_dim(row[1]) if len(row) > 1 else None
    form = row[2] if len(row) > 2 else ("int" if dense_dim is not None else "omitted")
    if form not in _CALL_FORMS:
        raise ValueError(
            f"_to_sparse_csr benchmark call form {form!r}: expected one of {_CALL_FORMS}"
        )
    if form == "int":
        if dense_dim is None:
            raise ValueError(
                f"_to_sparse_csr benchmark case {row!r}: the 'int' call form needs a dense_dim"
            )
        dim = dense_dim
    else:
        if dense_dim is not None:
            raise ValueError(
                f"_to_sparse_csr benchmark dense_dim {dense_dim!r}: the '{form}' call form "
                "passes no value, so dense_dim must be null"
            )
        dim = 0
    if not 0 <= dim <= len(spec) - 2:
        raise ValueError(
            f"_to_sparse_csr benchmark case {row!r}: dense_dim {dim} is outside the "
            f"valid range [0, {len(spec) - 2}] for this shape"
        )
    batch = 1
    for size in spec[: len(spec) - dim - 2]:
        batch *= size
    if batch == 0:
        # Measured native error: 'to_sparse_csr: Expected product of batch
        # dimensions to be non-zero.' Only the batch axes are constrained; an
        # empty sparse extent or an empty dense tail converts fine.
        raise ValueError(
            f"_to_sparse_csr benchmark case {row!r}: the batch dimensions before "
            "the sparse matrix axes must have a non-zero product"
        )
    return spec, dense_dim, form


def _pattern_host(shape, dense_dim):
    """Dense operand with its own zero pattern, built entirely on the host.

    Index arithmetic stays on host int64 and the payload is float32, so the
    fixture needs neither device int64 support nor a device scatter kernel; the
    caller moves the result to the benchmark device and casts it to the test
    dtype.

    Batched CSR requires an equal number of specified elements per batch, so the
    row pattern repeats over every batch while the column offsets and payload
    values differ. Magnitudes stay in 1..9, which every consts.FLOAT_DTYPES entry
    represents exactly.
    """
    shape = tuple(shape)
    dense = torch.zeros(shape, dtype=torch.float32)
    rank = len(shape)
    if rank < 2 or dense_dim > rank - 2:
        return dense
    rows = shape[rank - dense_dim - 2]
    cols = shape[rank - dense_dim - 1]
    batches = 1
    for size in shape[: rank - dense_dim - 2]:
        batches *= size
    block = 1
    for size in shape[rank - dense_dim :] if dense_dim else ():
        block *= size
    counts = torch.tensor(_row_counts(rows, cols), dtype=torch.int64)
    total = int(counts.sum().item())
    if total == 0 or batches == 0 or cols == 0:
        return dense
    crow = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    row_of = torch.repeat_interleave(torch.arange(rows), counts)
    within = torch.arange(total) - torch.repeat_interleave(crow[:-1], counts)
    span = (cols - counts).clamp(min=1)
    offsets = (
        torch.arange(rows)[:, None] * 3 + torch.arange(batches)[None, :] * 5 + 1
    ) % span[:, None]
    col = offsets[row_of] + within[:, None]
    payload = torch.arange(block)
    batch = torch.arange(batches)
    mag = (
        (
            within[:, None, None] * 3
            + row_of[:, None, None] * 5
            + col[:, :, None]
            + batch[None, :, None] * 7
            + payload[None, None, :]
        )
        % 9
    ) + 1
    flat = (
        (
            batch[None, :, None] * rows * cols
            + row_of[:, None, None] * cols
            + col[:, :, None]
        )
        * block
        + payload[None, None, :]
    ).reshape(-1)
    dense.view(-1).index_put_((flat,), mag.to(torch.float32).reshape(-1))
    return dense


def _make_dense_pattern(shape, dense_dim, dtype, device):
    return _pattern_host(shape, dense_dim).to(device=device, dtype=dtype)


def _case_fn(shape, dtype):
    del dtype
    spec, dense_dim, form = _case_descriptor(shape)
    dim = dense_dim if form == "int" else 0
    yield base.BenchmarkCasePlan(
        shape={"input": spec},
        params={
            "dense_dim": dense_dim,
            "call_form": form,
            "sparse_form": "batched" if dim < len(spec) - 2 else "uniform",
        },
        builder_args=(spec, dense_dim, form),
    )


def _build_inputs_fn(plan, dtype, device):
    spec, dense_dim, form = plan.builder_args
    dim = dense_dim if form == "int" else 0
    inp = _make_dense_pattern(spec, dim, dtype, device)
    return inp, ({} if form == "omitted" else {"dense_dim": dense_dim})


class ToSparseCsrBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # A configured shape file is honoured through the framework loader; only
        # the fallback descriptors are replaced by this operator's workloads.
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark.to_sparse_csr
def test__to_sparse_csr():
    bench = ToSparseCsrBenchmark(
        op_name="_to_sparse_csr",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._to_sparse_csr,
        gems_op=getattr(flag_gems, "_to_sparse_csr", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
