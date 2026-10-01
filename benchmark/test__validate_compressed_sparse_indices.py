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

"""Benchmark for ``aten::_validate_compressed_sparse_indices``.

``shape[-1]`` is the requested compressed-dimension size ``cdim`` and the
leading entries are batch dimensions, so one plan describes ``compressed_idx``
of shape ``batch + (cdim + 1,)`` and ``plain_idx`` of shape ``batch + (nnz,)``
in the plan's index dtype.  ``_row_lengths`` fills the compressed rows with at
most 8 entries each and the rows it does not reach stay empty, so the row count
is exactly ``cdim`` and the entry count is exactly ``nnz`` -- the listed
geometry and the materialized tensors always agree.  Case plans stay tensor-free
JSON metadata, so --list-cases works before a candidate exists and replays
exactly the plans that timing uses; the tensors are built only by the input
builder.
"""

import itertools
import numbers

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# int32/int64 only: AT_DISPATCH_INDEX_TYPES rejects every other index dtype.
_INDEX_DTYPES = [torch.int32]
if flag_gems.runtime.device.support_int64:
    _INDEX_DTYPES.append(torch.int64)

# The original descriptor scales: every entry but the last is a batch dimension
# and the last one is the compressed-dimension size cdim.
_BENCH_SHAPES = ((1024, 1024), (20, 320, 15), (8, 16, 512))

# The remaining original scales.  They belong to the operator's own fallback
# geometry list rather than being injected for one benchmark level only, so
# listing and execution always work on the same five descriptors.
_EXTRA_SHAPES = ((256,), (64, 1024))

_DEFAULT_SHAPES = _BENCH_SHAPES + _EXTRA_SHAPES


def _row_lengths(cdim):
    """Compressed row lengths holding at most 8 plain entries per row.

    A row's plain indices are ``arange(count)``, so the longest row bounds the
    largest index; keeping it at 8 or below lets ``dim`` stay small for every
    default shape.  The lengths sum to ``cdim`` and the caller pads the list to
    exactly ``cdim`` rows, so the requested compressed dimension is never
    shrunk to the number of filled rows.
    """
    if cdim <= 0:
        return ()
    block = min(8, cdim)
    return (block,) * (cdim // block) + ((cdim % block,) if cdim % block else ())


def _validated_shape(shape):
    """Exact non-negative integer extents, without coercion or clamping.

    A float or bool extent, a non-sequence, or a negative extent is a malformed
    shape request and is rejected instead of being truncated into a different
    descriptor.  Zero extents stay valid.
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"a benchmark shape must be a sequence, got {shape!r}")
    extents = []
    for entry in shape:
        if isinstance(entry, bool) or not isinstance(entry, numbers.Integral):
            raise TypeError(f"shape extents must be integers, got {entry!r}")
        if entry < 0:
            raise ValueError(f"shape extents must be non-negative, got {entry!r}")
        extents.append(int(entry))
    if not extents:
        raise ValueError("a compressed descriptor needs at least one dimension")
    return tuple(extents)


def _case_fn(shape, dtype):
    shape = _validated_shape(shape)
    batch, cdim = shape[:-1], shape[-1]
    filled = _row_lengths(cdim)
    # The filled rows plus the padding rows are exactly ``cdim`` compressed rows,
    # so cdim and nnz keep their original values and the metadata below matches
    # what the builder materializes.  They are equal because the row lengths sum
    # to cdim (0 when cdim is 0).
    counts = filled + (0,) * (cdim - len(filled))
    nnz = sum(counts)
    # cdim == 0 still needs a legal non-negative plain-dimension size.
    dim = min(8, cdim) or 1
    for is_crow in (False, True):
        yield base.BenchmarkCasePlan(
            shape={
                "batch": list(batch),
                "cdim": cdim,
                "compressed": [*batch, cdim + 1],
                "plain": [*batch, nnz],
            },
            params={
                "is_crow": is_crow,
                "cdim": cdim,
                "nnz": nnz,
                "dim": dim,
                "index_dtype": str(dtype).removeprefix("torch."),
            },
            builder_args=(batch, cdim, counts, dim, is_crow),
        )


def _build_inputs_fn(plan, dtype, device):
    # The plan's dtype is the index dtype of both tensors, and the tensors must
    # live on the framework device.
    batch, cdim, counts, dim, is_crow = plan.builder_args
    nnz = sum(counts)
    # Offsets are the prefix sums of the row lengths and every row's indices are
    # ``arange(count)``, so the leading offset is 0, the offsets are monotone and
    # the terminal offset is exactly ``nnz``.
    compressed = torch.tensor(
        [0, *itertools.accumulate(counts)], dtype=dtype, device=device
    )
    plain = torch.tensor(
        [value for count in counts for value in range(count)],
        dtype=dtype,
        device=device,
    )
    if batch:
        compressed = compressed.expand(*batch, -1).contiguous()
        plain = plain.expand(*batch, -1).contiguous()
    # Positional native signature: (is_crow, cidx, idx, cdim, dim, nnz).
    return (is_crow, compressed, plain, cdim, dim, nnz)


class ValidateCompressedSparseIndicesBenchmark(OperatorBenchmark):
    """Descriptor-validator benchmark on the operator's own geometries."""

    def set_shapes(self, shape_file_path=None):
        # The generic core_shapes.yaml entries are far too large once their last
        # dimension is read as a compressed dimension, so this operator's own
        # geometry list is the fallback.  The established OperatorBenchmark
        # precedence is kept: an operator- or class-keyed shape file overrides
        # the fallback, and an explicitly empty ``shapes`` list stays empty.
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)


@pytest.mark.validate_compressed_sparse_indices
def test_validate_compressed_sparse_indices():
    bench = ValidateCompressedSparseIndicesBenchmark(
        op_name="_validate_compressed_sparse_indices",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._validate_compressed_sparse_indices,
        gems_op=getattr(flag_gems, "_validate_compressed_sparse_indices", None),
        dtypes=_INDEX_DTYPES,
    )
    bench.run()
