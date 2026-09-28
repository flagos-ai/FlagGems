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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# Performance scales. Every scale is at least 2-D because to_sparse_csr converts
# a matrix or a batch of matrices.
TO_SPARSE_CSR_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

# (input shape, requested dense_dim, payload pattern) plans. dense_dim selects
# how many trailing axes become the dense tail: 0 leaves exactly the two sparse
# axes (any leading axis is then a batch axis) and rank - 2 makes every trailing
# axis dense. "dense" is a random fill, "structured" stores a diagonal plus two
# columns, "batched" additionally rotates and rescales every batch, and "tail"
# additionally varies the dense tail, so a repeated batch or a repeated tail
# cannot match.
TO_SPARSE_CSR_CASES = [
    ((1024, 1024), 0, "dense"),
    ((1024, 1024), 0, "structured"),
    ((4096, 4096), 0, "dense"),
    ((4096, 4096), 0, "structured"),
    ((20, 320, 15), 0, "dense"),
    ((20, 320, 15), 0, "structured"),
    ((20, 320, 15), 0, "batched"),
    ((20, 320, 15), 1, "dense"),
    ((20, 320, 15), 1, "structured"),
    ((20, 320, 15), 1, "tail"),
    ((16, 128, 64, 60), 0, "batched"),
    ((16, 128, 64, 60), 1, "dense"),
    ((16, 128, 64, 60), 1, "structured"),
    ((16, 128, 64, 60), 1, "tail"),
    ((16, 128, 64, 60), 2, "structured"),
    ((16, 128, 64, 60), 2, "tail"),
    ((16, 7, 57, 32, 29), 0, "batched"),
    ((16, 7, 57, 32, 29), 1, "dense"),
    ((16, 7, 57, 32, 29), 1, "structured"),
    ((16, 7, 57, 32, 29), 1, "tail"),
    ((16, 7, 57, 32, 29), 2, "structured"),
    ((16, 7, 57, 32, 29), 3, "structured"),
]

_PATTERNS = ("dense", "structured", "batched", "tail")
_TAIL_POSITIONS = 4
_TAIL_BATCHES = 4


def _bench_dtypes():
    # Static device capability metadata, no tensor allocated here: bf16 stays
    # out on a backend that cannot run it.
    if flag_gems.runtime.device.support_bf16:
        return list(consts.FLOAT_DTYPES)
    return [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16]


_DTYPES = _bench_dtypes()


def _checked_shape(shape):
    """Validate one requested input shape and return it as a tuple.

    Only a matrix or a batch of matrices with non-negative integer extents is a
    valid request; a zero extent (a zero matrix or a zero dense tail) stays
    valid, and the batch-product restriction is checked separately once the
    requested dense_dim is known. An invalid entry raises here, before any
    input tensor is allocated, instead of being dropped silently.
    """
    if isinstance(shape, bool) or not isinstance(shape, (tuple, list)):
        raise ValueError(f"to_sparse_csr needs a shape sequence, got {shape!r}")
    dims = tuple(shape)
    if len(dims) < 2:
        raise ValueError(f"to_sparse_csr needs at least 2 dimensions, got {dims}")
    for extent in dims:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ValueError(f"to_sparse_csr extents must be integers, got {dims}")
        if extent < 0:
            raise ValueError(f"to_sparse_csr extents must be non-negative, got {dims}")
    return dims


def _split(shape, dense_dim):
    """Batch axes, dense-tail axes and the two sparse axes of one request."""
    start = len(shape) - 2 - dense_dim
    return (
        tuple(shape[:start]),
        tuple(shape[start + 2 :]),
        (shape[start], shape[start + 1]),
    )


def _check_batch_product(shape, dense_dim):
    """Reject a batched request whose batch product is zero.

    Measured natively: (0, 3, 4) at dense_dim 0 and (2, 0, 3, 4) at dense_dim 0
    raise "Expected product of batch dimensions to be non-zero.", and (0, 0, 3,
    4) at dense_dim 1 does too, while a zero matrix or a zero dense tail next to
    a non-zero batch product ((0, 3, 4) at dense_dim 1, (0, 0), (2, 3, 0))
    stays valid. The check uses only the requested metadata, so it runs before
    any input tensor is allocated.
    """
    batch_shape = _split(shape, dense_dim)[0]
    if math.prod(batch_shape) == 0:
        raise ValueError(
            f"to_sparse_csr needs a non-zero batch product for {shape} at "
            f"dense_dim {dense_dim}, got batch axes {batch_shape}"
        )


def _checked_descriptor(shape, dense_dim, pattern):
    shape = _checked_shape(shape)
    if isinstance(dense_dim, bool) or not isinstance(dense_dim, int):
        raise ValueError(
            f"to_sparse_csr dense_dim must be an integer, got {dense_dim!r}"
        )
    if dense_dim < 0 or dense_dim > len(shape) - 2:
        raise ValueError(
            f"to_sparse_csr dense_dim must be in [0, {len(shape) - 2}] for {shape}"
        )
    if pattern not in _PATTERNS:
        raise ValueError(
            f"unknown payload pattern {pattern!r}, expected one of {_PATTERNS}"
        )
    _check_batch_product(shape, dense_dim)
    return shape, dense_dim, pattern


def _requested_descriptor(entry):
    """Return an explicit (shape, dense_dim, pattern) request, or None.

    None means the entry is a plain shape, for which the schema default
    dense_dim (0) applies.
    """
    if (
        isinstance(entry, (tuple, list))
        and len(entry) == 3
        and isinstance(entry[0], (tuple, list))
    ):
        return _checked_descriptor(entry[0], entry[1], entry[2])
    return None


def _case_rows(shape):
    """The plans for one entry of the requested shape list."""
    requested = _requested_descriptor(shape)
    if requested is not None:
        return [requested]
    shape = _checked_shape(shape)
    # A plain entry uses the schema default dense_dim 0, so exactly the two
    # trailing axes stay sparse and every leading axis is a batch axis.
    _check_batch_product(shape, 0)
    rows = [row for row in TO_SPARSE_CSR_CASES if row[0] == shape]
    if rows:
        return rows
    # A caller-supplied scale keeps the schema default dense_dim (0, so exactly
    # the two trailing axes stay sparse) with the low-density fill and the
    # random fill; the requested shape itself is used unchanged.
    return [(shape, 0, "structured"), (shape, 0, "dense")]


def _case_fn(shape, dtype):
    del dtype
    for case_shape, dense_dim, pattern in _case_rows(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": case_shape},
            params={"dense_dim": dense_dim, "pattern": pattern},
            builder_args=(case_shape, dense_dim, pattern),
        )


def _matrix_block(shape2d, pattern, dtype, device):
    """The 2-D payload of one sparse plane."""
    if pattern == "dense":
        return utils.generate_tensor_input(shape2d, dtype, device)
    rows, cols = shape2d
    block = torch.zeros(shape2d, dtype=dtype, device=device)
    if rows == 0 or cols == 0:
        return block
    # A diagonal plus two columns: few stored entries, at least one per row, and
    # no auxiliary index tensor of any dtype.
    block.diagonal().fill_(1.0)
    block[:, 0] = 2.0
    if cols > 1:
        block[:, cols - 1] = 3.0
    return block


def _roll_planes(block, batch_count, dtype, device):
    """Rotate the columns and rescale the values of every batch.

    The rotation moves stored entries to other columns without changing how many
    entries a row has, and the strictly positive scale keeps them stored, so each
    batch still converts with the same number of stored elements (the native
    requirement for a batched input) at different positions and values. Slicing
    and torch.cat keep the fixture free of index tensors; the operator's own
    int64 index contract is not an input-builder concern.
    """
    rows, cols = block.shape
    if batch_count < 2 or rows == 0 or cols == 0:
        return block.reshape((1, rows, cols)).expand(batch_count, rows, cols)
    planes = []
    for index in range(batch_count):
        shift = index % cols
        if shift == 0:
            planes.append(block)
        else:
            planes.append(
                torch.cat((block[:, cols - shift :], block[:, : cols - shift]), dim=1)
            )
    scale = torch.tensor(
        [float(index + 1) for index in range(batch_count)],
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    return torch.stack(planes) * scale.reshape(batch_count, 1, 1)


def _tail_factor(batch_shape, batch_count, tail_numel, dtype, device):
    """Strictly positive, nonuniform values for the dense tail.

    The dense tail of a single matrix would otherwise repeat one value, so the
    factors vary both along the tail and from batch to batch. Strictly positive
    factors keep every stored element stored, which is what makes a batched
    conversion legal.
    """
    positions = torch.arange(tail_numel, dtype=torch.float32, device=device)
    ramp = ((positions % _TAIL_POSITIONS) + 1.0).reshape(
        (1,) * len(batch_shape) + (1, tail_numel)
    )
    factors = torch.tensor(
        [float(1 + (index % _TAIL_BATCHES)) for index in range(batch_count)],
        dtype=torch.float32,
        device=device,
    ).reshape(batch_shape + (1, 1))
    return (ramp * factors).to(dtype)


def _dense_input(shape, dense_dim, pattern, dtype, device):
    batch_shape, tail_shape, (rows, cols) = _split(shape, dense_dim)
    batch_count = math.prod(batch_shape) if batch_shape else 1
    block = _matrix_block((rows, cols), pattern, dtype, device)
    if pattern == "batched":
        planes = _roll_planes(block, batch_count, dtype, device)
    else:
        planes = block.reshape((1,) * len(batch_shape) + (rows, cols)).expand(
            batch_shape + (rows, cols)
        )
    planes = planes.reshape(batch_shape + (rows, cols))
    if not tail_shape:
        return planes.contiguous()
    tail_numel = math.prod(tail_shape)
    tail = planes.reshape(batch_shape + (rows * cols, 1)).expand(
        batch_shape + (rows * cols, tail_numel)
    )
    if pattern == "tail":
        tail = tail * _tail_factor(batch_shape, batch_count, tail_numel, dtype, device)
    return tail.reshape(shape).contiguous()


def _build_inputs_fn(plan, dtype, device):
    shape, dense_dim, pattern = plan.builder_args
    inp = _dense_input(shape, dense_dim, pattern, dtype, device)
    # The requested dense_dim is the operator's second positional argument,
    # exactly as the plan params record it.
    return inp, dense_dim


class ToSparseCsrBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # The shared OperatorBenchmark resolver supplies the five scales above
        # (or the caller's shape file) and merges no extra extension shapes for
        # this operator. Every requested entry is validated here, before any
        # input tensor is allocated: an invalid descriptor raises instead of
        # being dropped, a zero extent stays valid, and a zero batch product is
        # rejected after the dense_dim split.
        super().set_shapes(shape_file_path, default_shapes=TO_SPARSE_CSR_SHAPES)
        for entry in self.shapes:
            _case_rows(entry)


@pytest.mark.to_sparse_csr
def test_to_sparse_csr():
    bench = ToSparseCsrBenchmark(
        op_name="to_sparse_csr",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.to_sparse_csr,
        gems_op=getattr(flag_gems, "to_sparse_csr", None),
        dtypes=_DTYPES,
    )
    bench.run()
