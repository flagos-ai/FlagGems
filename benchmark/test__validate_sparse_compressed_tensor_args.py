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

from typing import NamedTuple

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# aten::_validate_sparse_compressed_tensor_args(compressed_indices, plain_indices,
#     values, int[] size, Layout layout) -> ()
#
# The measured work is argument dispatch plus the host-side structural checks and
# the device-side index scan over nnz, so a benchmark case is a full compressed
# tensor description rather than a plain tensor shape. A serialised row is
#     (layout, batch extents, base extents, dense extents, block shape, nnz,
#      index dtype)
# and the logical size handed to the op is batch + (rows, cols) + dense. Layout and
# index dtype are documented string keys so a shape file can carry them; the
# normaliser decodes them and rejects anything malformed before a tensor is
# allocated. A zero compressed extent only accepts nnz == 0 (nothing can be stored
# there and nnz must never be divided by it); a zero-size block is rejected as
# well, because the native op asserts blocksize > 0 with TORCH_INTERNAL_ASSERT.
#
# Broadcast does not apply (extents are checked against `size`, never broadcast
# between operands) and backward does not apply (the schema returns ()).

_LAYOUT_KEYS = {
    "csr": torch.sparse_csr,
    "csc": torch.sparse_csc,
    "bsr": torch.sparse_bsr,
    "bsc": torch.sparse_bsc,
}
_ROW_MAJOR_LAYOUTS = (torch.sparse_csr, torch.sparse_bsr)
_BLOCKED_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
_INDEX_KEYS = {"int32": torch.int32, "int64": torch.int64}
_INDEX_LIMITS = {torch.int32: 2**31 - 1, torch.int64: 2**63 - 1}

# Static capability flags, not runtime probes: the values component only has to be a
# strided tensor of the right shape, so a value dtype is dropped only when the
# backend cannot materialise it at all, and an int64 index description is dropped
# only without int64 support. The int32 descriptions are independent of both.
_INT64_SUPPORTED = flag_gems.runtime.device.support_int64
_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


class _Description(NamedTuple):
    """One normalised description row."""

    layout: torch.layout
    batch: tuple
    base: tuple
    dense: tuple
    blocks: tuple
    nnz: int
    index_dtype: torch.dtype


def _compressed_dim(description):
    rows, cols = description.base
    block_rows, block_cols = description.blocks
    if description.layout in _ROW_MAJOR_LAYOUTS:
        return rows // block_rows
    return cols // block_cols


def _plain_dim(description):
    rows, cols = description.base
    block_rows, block_cols = description.blocks
    if description.layout in _ROW_MAJOR_LAYOUTS:
        return cols // block_cols
    return rows // block_rows


def _block_shape(description):
    return description.blocks if description.layout in _BLOCKED_LAYOUTS else ()


def _logical_size(description):
    return [*description.batch, *description.base, *description.dense]


def _compressed_shape(description):
    return [*description.batch, _compressed_dim(description) + 1]


def _plain_shape(description):
    return [*description.batch, description.nnz]


def _values_shape(description):
    return [
        *description.batch,
        description.nnz,
        *_block_shape(description),
        *description.dense,
    ]


def _extents(name, value):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of extents, got {value!r}")
    extents = []
    for extent in value:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ValueError(f"{name} extent must be an int, got {extent!r}")
        if extent < 0:
            raise ValueError(f"{name} extent must be non-negative, got {extent}")
        extents.append(int(extent))
    return tuple(extents)


def normalize_description(row):
    """Validate and normalise one serialised description row.

    Every structural precondition is checked here, before any tensor exists: row
    arity, integer (non-bool) non-negative extents and nnz, positive block extents,
    block divisibility, index representability and the capacity
    ceil(nnz / compressed_dim) <= plain_dim. A zero compressed extent accepts only
    nnz == 0; a requested nnz is never clamped or recomputed.
    """
    if not isinstance(row, (list, tuple)) or len(row) != 7:
        raise ValueError(f"description row must have 7 entries, got {row!r}")
    layout_key, batch, base, dense, blocks, nnz, index_key = row
    layout_key = str(layout_key).removeprefix("torch.sparse_").lower()
    if layout_key not in _LAYOUT_KEYS:
        raise ValueError(f"unsupported layout key: {layout_key!r}")
    if isinstance(index_key, torch.dtype):
        index_dtype = index_key
    else:
        index_dtype = _INDEX_KEYS.get(str(index_key).removeprefix("torch.").lower())
    if index_dtype not in _INDEX_LIMITS:
        raise ValueError(f"unsupported index dtype: {index_key!r}")
    batch = _extents("batch", batch)
    base = _extents("base", base)
    dense = _extents("dense", dense)
    if len(base) != 2:
        raise ValueError(f"base must carry 2 extents, got {base!r}")
    raw_blocks = _extents("blocks", blocks)
    if len(raw_blocks) != 2:
        raise ValueError(f"block shape must carry 2 extents, got {raw_blocks!r}")
    if raw_blocks[0] < 1 or raw_blocks[1] < 1:
        raise ValueError(f"block extents must be positive, got {raw_blocks!r}")
    if isinstance(nnz, bool) or not isinstance(nnz, int):
        raise ValueError(f"nnz must be an int, got {nnz!r}")
    if nnz < 0:
        raise ValueError(f"nnz must be non-negative, got {nnz}")
    description = _Description(
        _LAYOUT_KEYS[layout_key],
        batch,
        base,
        dense,
        raw_blocks,
        int(nnz),
        index_dtype,
    )
    if description.layout in _BLOCKED_LAYOUTS:
        if base[0] % raw_blocks[0] or base[1] % raw_blocks[1]:
            raise ValueError(f"base {base} is not divisible by blocks {raw_blocks}")
    elif raw_blocks != (1, 1):
        raise ValueError(f"plain layout {layout_key} has no block shape")

    limit = _INDEX_LIMITS[index_dtype]
    for extent in (
        _compressed_dim(description) + 1,
        _plain_dim(description),
        description.nnz,
    ):
        if extent > limit:
            raise ValueError(f"description does not fit the {index_dtype} index range")
    compressed_dim = _compressed_dim(description)
    if compressed_dim == 0:
        if description.nnz:
            raise ValueError(
                f"a zero compressed extent cannot store nnz={description.nnz}"
            )
    else:
        plain_dim = _plain_dim(description)
        if -(-description.nnz // compressed_dim) > plain_dim:
            raise ValueError(
                f"nnz={description.nnz} exceeds the {compressed_dim} x "
                f"{plain_dim} capacity"
            )
    return description


# (layout, batch extents, base extents, dense extents, block shape, nnz, index dtype)
_BENCH_CASES = [
    ("csr", (), (1024, 1024), (), (1, 1), 16384, "int32"),
    ("csr", (), (1024, 1024), (), (1, 1), 16384, "int64"),
    ("csc", (), (1024, 1024), (), (1, 1), 16384, "int32"),
    ("bsr", (), (1024, 1024), (), (2, 2), 8192, "int32"),
    ("bsc", (), (1024, 1024), (), (2, 2), 8192, "int32"),
    ("csr", (), (4096, 4096), (), (1, 1), 65536, "int64"),
    ("csr", (), (20, 320), (), (1, 1), 3200, "int32"),
    ("csc", (), (20, 320), (), (1, 1), 3200, "int32"),
    ("bsr", (), (20, 320), (), (2, 2), 1600, "int32"),
    ("csr", (4,), (16, 128), (), (1, 1), 512, "int32"),
    ("csr", (), (16, 128), (4,), (1, 1), 1024, "int32"),
    # Required canonical logical sizes, expressed as batch + base + dense.
    ("csr", (16,), (128, 64), (60,), (1, 1), 4096, "int32"),
    ("csc", (16,), (128, 64), (60,), (1, 1), 4096, "int32"),
    ("bsr", (16,), (128, 64), (60,), (2, 2), 2048, "int32"),
    ("bsc", (16,), (128, 64), (60,), (2, 2), 2048, "int32"),
    ("csr", (16, 7), (57, 32), (29,), (1, 1), 500, "int32"),
    ("bsr", (16, 7), (57, 32), (29,), (1, 1), 500, "int32"),
    ("csr", (20,), (320, 15), (), (1, 1), 1200, "int32"),
    ("bsr", (), (20, 320), (15,), (2, 2), 1600, "int32"),
    ("csr", (2, 3), (8, 10), (), (1, 1), 9, "int32"),
    # Empty descriptions with a zero compressed extent: valid, and no division by it.
    ("csr", (), (0, 4), (), (1, 1), 0, "int32"),
    ("bsc", (), (0, 0), (), (1, 1), 0, "int32"),
]


def _default_descriptions():
    descriptions = []
    for row in _BENCH_CASES:
        description = normalize_description(row)
        if description.index_dtype is torch.int64 and not _INT64_SUPPORTED:
            # A backend without int64 cannot materialise these index tensors.
            continue
        descriptions.append(description)
    return tuple(descriptions)


_DEFAULT_DESCRIPTIONS = _default_descriptions()


def _per_batch_indices(description, batch_index):
    """One batch's (compressed row offsets, plain indices).

    Counts are spread over the compressed dimension and each batch is rotated, so
    the description holds independent valid patterns per batch, and every row's
    plain indices are a sorted run of in-range positions as the device-side scan
    requires.
    """
    compressed_dim = _compressed_dim(description)
    plain_dim = _plain_dim(description)
    compressed_row = [0]
    plain = []
    if compressed_dim:
        quotient, remainder = divmod(description.nnz, compressed_dim)
        for row in range(compressed_dim):
            count = quotient + (
                1 if ((row + batch_index) % compressed_dim) < remainder else 0
            )
            compressed_row.append(compressed_row[-1] + count)
            if count:
                start = ((batch_index + 1) * 7 + row * 3) % (plain_dim - count + 1)
                plain.extend(range(start, start + count))
    return compressed_row, plain


def _build_description(description, dtype, device):
    batch_count = 1
    for extent in description.batch:
        batch_count *= extent
    compressed_flat = []
    plain_flat = []
    for batch_index in range(batch_count):
        compressed_row, plain = _per_batch_indices(description, batch_index)
        compressed_flat.extend(compressed_row)
        plain_flat.extend(plain)
    compressed = torch.tensor(
        compressed_flat, dtype=description.index_dtype, device=device
    ).reshape(*_compressed_shape(description))
    plain = torch.tensor(
        plain_flat, dtype=description.index_dtype, device=device
    ).reshape(*_plain_shape(description))
    values = utils.generate_tensor_input(
        tuple(_values_shape(description)), dtype, device
    )
    return compressed, plain, values, _logical_size(description)


def _case_fn(shape, dtype):
    # `shape` is one entry of the default description list (a normalised
    # _Description) or, for an operator shape file, the same serialised row. Only
    # the plan is produced here, so --list-cases allocates no tensor.
    description = (
        shape if isinstance(shape, _Description) else normalize_description(shape)
    )
    yield base.BenchmarkCasePlan(
        shape={
            "logical_size": _logical_size(description),
            "compressed_indices_shape": _compressed_shape(description),
            "plain_indices_shape": _plain_shape(description),
            "values_shape": _values_shape(description),
            "block_shape": list(description.blocks),
        },
        params={
            "layout": str(description.layout),
            "index_dtype": str(description.index_dtype),
            "values_dtype": str(dtype),
            "nnz": description.nnz,
            "batch_ndim": len(description.batch),
            "dense_ndim": len(description.dense),
        },
        builder_args=(description,),
    )


def _build_inputs_fn(plan, dtype, device):
    description = plan.builder_args[0]
    compressed, plain, values, logical_size = _build_description(
        description, dtype, device
    )
    # The trailing dict becomes keyword arguments: Benchmark.unpack_to_args_kwargs
    # only forwards tensors, numbers, strings, None, lists/tuples and dtypes, so a
    # positional torch.layout would be dropped and the layout would be lost.
    return compressed, plain, values, logical_size, {"layout": description.layout}


class ValidateSparseCompressedTensorArgsBenchmark(OperatorBenchmark):
    """Benchmark over compressed-tensor descriptions instead of plain shapes."""

    def set_shapes(self, shape_file_path=None):
        # A shape file may still carry an entry for this operator; its rows are the
        # same serialised descriptions accepted by normalize_description. No extra
        # shapes are merged here: OperatorBenchmark takes `shapes` from that file or
        # from these defaults, and the pointwise shapes a plain GenericBenchmark
        # would add cannot describe a compressed tensor.
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_DESCRIPTIONS)


@pytest.mark.validate_sparse_compressed_tensor_args
def test_validate_sparse_compressed_tensor_args():
    bench = ValidateSparseCompressedTensorArgsBenchmark(
        op_name="_validate_sparse_compressed_tensor_args",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._validate_sparse_compressed_tensor_args,
        gems_op=getattr(flag_gems, "_validate_sparse_compressed_tensor_args", None),
        dtypes=_DTYPES,
    )
    bench.run()
