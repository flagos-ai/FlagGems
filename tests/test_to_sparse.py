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

# Correctness tests for aten::to_sparse (value-preserving layout conversion).
#
# Both invocable call forms are reached through the single public candidate
# flag_gems.to_sparse: the positional sparse_dim overload and the keyword form
# (layout / blocksize / dense_dim). Querying torch.ops.aten.to_sparse.out
# reports that the underlying op has no overload name out, and a single tensor
# operand cannot broadcast, so the spec broadcast dimension does not apply.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

_COMPRESSED_LAYOUTS = (
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
)
_BLOCK_LAYOUTS = (torch.sparse_bsr, torch.sparse_bsc)
# An omitted layout target is the schema default, which is coalesced COO.
_COO_TARGETS = (None, torch.sparse_coo)


def _device_supports(dtype):
    # Static capability lookup: no native probe runs at collection or execution
    # time, the flags are read from the active backend once at import.
    if dtype in (torch.float64, torch.complex128):
        return utils.fp64_is_supported
    if dtype == torch.int64:
        return utils.int64_is_supported
    if dtype == torch.bfloat16:
        return utils.bf16_is_supported
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return utils.fp8_is_supported
    return True


_MAIN_DTYPES = [
    dtype
    for dtype in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    if _device_supports(dtype)
]
_EXTRA_DTYPES = [torch.float32, torch.int32]
_LAYOUT_DTYPES = [torch.float32] + (
    [torch.bfloat16] if _device_supports(torch.bfloat16) else []
)
_BACKWARD_DTYPES = [torch.float32, torch.float16]
# complex64 is preserved independently; complex128 needs 64-bit floating point
# kernels, so it is gated by the same capability flag as float64.
_COMPLEX_DTYPES = (torch.complex64,) + (
    (torch.complex128,) if _device_supports(torch.complex128) else ()
)

# Native probe on this backend: the dense to COO conversion, and every form that
# has to materialise COO indices (including dense_dim), run the nonzero kernel,
# which has no FP8 instantiation:
#     RuntimeError: "nonzero_cuda" not implemented for 'Float8E4m3fn'
# (same message for Float8E5m2). layout=torch.strided and the compressed
# layouts build their result from the stored values instead and accept both FP8
# dtypes, so FP8 keeps positive coverage on the compressed and strided rows and
# its COO forms are negative rows instead of being replaced by another dtype.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2) if utils.fp8_is_supported else ()
_FP8_COO_UNSUPPORTED_DTYPES = [
    dtype for dtype in _FP8_DTYPES if _device_supports(dtype)
]

# A COO result carries an int64 index tensor whatever the payload dtype is, and
# the already-sparse sources and the backward upstreams carry int64 index
# tensors as well, so every family whose comparison includes that structure is
# gated on the backend's static int64 support. The compressed layouts keep the
# index dtype they were built with, and the strided identity has no index tensor
# at all, so those paths stay enabled independently of this flag.
_COO_STRUCTURE_OK = utils.int64_is_supported


def _equal_count_source(inp, layout, blocksize, dense_dim=None):
    # Batched CSR/CSC/BSR/BSC reject an unequal per-batch stored count (native
    # message: Expect the same number of specified elements per batch.), and a
    # fixed position mask is not safe because generated low precision values can
    # round to an exact zero. The mask is therefore derived from the tensor own
    # nonzero pattern: every batch keeps its first keep grid positions (blocks
    # for the block layouts), where keep is the smallest per-batch count, so all
    # batches store the same number of entries while keeping their own positions
    # and values. A kernel that replicates batch 0 still fails the comparison.
    # dense_dim moves the matrix dimensions away from the last two axes, so the
    # dense tail is collapsed before the per-batch grid is built.
    dense_dim = 0 if dense_dim is None else int(dense_dim)
    core_len = inp.dim() - dense_dim
    core_shape = inp.shape[:core_len]
    rows, cols = core_shape[-2], core_shape[-1]
    batch_shape = core_shape[:-2]
    zero = torch.zeros((), dtype=inp.dtype, device=inp.device)
    nonzero = inp != zero
    if dense_dim:
        nonzero = nonzero.any(dim=tuple(range(core_len, inp.dim())))
    if layout in _BLOCK_LAYOUTS:
        grid_rows = rows // blocksize[0]
        grid_cols = cols // blocksize[1]
        blocked = nonzero.reshape(
            batch_shape + (grid_rows, blocksize[0], grid_cols, blocksize[1])
        )
        grid = blocked.any(-1).any(-2)
    else:
        grid_rows = rows
        grid_cols = cols
        grid = nonzero.reshape(batch_shape + (grid_rows, grid_cols))
    batches = 1
    for size in batch_shape:
        batches = batches * size
    flat = grid.reshape(batches, grid_rows * grid_cols)
    # The per-batch grid holds rows*cols positions at most (1048576 for the
    # widest compressed row here, 1024x1024), which stays far below the int32
    # range, so int32 is the adequate accumulator for both the bounded per-batch
    # count and the running rank; a wider dtype is not required and would only
    # change the dtype of the temporary mask arithmetic.
    keep = int(flat.sum(-1, dtype=torch.int32).min().item())
    rank = torch.cumsum(flat.to(torch.int32), dim=-1, dtype=torch.int32)
    kept = flat & (rank <= keep)
    mask = kept.reshape(batch_shape + (grid_rows, grid_cols))
    if layout in _BLOCK_LAYOUTS:
        mask = mask.repeat_interleave(blocksize[0], -2)
        mask = mask.repeat_interleave(blocksize[1], -1)
    if dense_dim:
        for _ in range(dense_dim):
            mask = mask.unsqueeze(-1)
        mask = mask.expand(inp.shape)
    return torch.where(mask, inp, zero)


def _masked_input(dtype, shape, value_range, layout, blocksize, dense_dim=None):
    # Dense source for the target layout; a batched compressed target needs the
    # equal per-batch stored count described above.
    inp = tu.make_input(dtype, shape, value_range)
    if layout in _COMPRESSED_LAYOUTS and len(shape) > 2:
        inp = _equal_count_source(inp, layout, blocksize, dense_dim)
    return inp


def _view_input(dtype, shape, variant, value_range, layout=None, blocksize=None):
    # Returns the larger allocation and the dense view of it: nonzero storage
    # offset, transposed view and stride 0 view. The parent is returned because
    # the view shares its storage and cannot expose writes into the rows that
    # the view does not reach.
    parent = _masked_input(dtype, shape, value_range, layout, blocksize)
    if variant == "contig":
        return parent, parent
    if variant == "offset":
        return parent, parent[1:]
    if variant == "permuted":
        return parent, parent.transpose(-1, -2)
    return parent, parent[:, :1].expand(shape)


def _layout_kwargs(layout, blocksize, dense_dim=None):
    kwargs = {}
    if layout is not None:
        kwargs["layout"] = layout
    if blocksize is not None:
        kwargs["blocksize"] = [int(blocksize[0]), int(blocksize[1])]
    if dense_dim is not None:
        kwargs["dense_dim"] = int(dense_dim)
    return kwargs


# Main spec grid: the seven spec shapes with the full dtype list, plus a few
# additional ranks and extents (rank 1-4 at odd, non-square or small extents)
# that keep the coordinate arithmetic away from the spec extents. Every entry
# uses the positional sparse_dim overload with sparse_dim = min(2, rank).
_SPEC_SHAPE_CASES = [
    (shape, dtype) for shape in tu.selected_shapes() for dtype in _MAIN_DTYPES
]
_EXTRA_SHAPES = [(128,), (64, 64), (8, 16, 32), (2, 3, 4, 5), (9,), (7, 11, 13)]
_EXTRA_SHAPE_CASES = [
    (shape, dtype)
    for shape in tu.selected_cases(_EXTRA_SHAPES, quick=[])
    for dtype in _EXTRA_DTYPES
]
_GRID_CASES = (_SPEC_SHAPE_CASES + _EXTRA_SHAPE_CASES) if _COO_STRUCTURE_OK else []

# sparse_dim overload at interior ranks, including 0 and rank-1 (fully sparse)
# and the hybrid forms that keep a dense tail. The small-shape rows are part of
# the default table only, so the default suite keeps them while quick mode
# stays on the main spec grid.
_SMALL_SPARSE_DIM_ROWS = [((2, 19, 7), 2)]
_SPARSE_DIM_ROWS = (
    (
        [
            ((), 0),
            ((1,), 1),
            ((256,), 1),
            ((1024, 1024), 1),
            ((1024, 1024), 2),
            ((20, 320, 15), 1),
            ((20, 320, 15), 2),
            ((20, 320, 15), 3),
            ((16, 128, 64, 60), 2),
            ((16, 128, 64, 60), 3),
            ((16, 7, 57, 32, 29), 2),
            ((16, 7, 57, 32, 29), 3),
            ((16, 7, 57, 32, 29), 4),
        ]
        + _SMALL_SPARSE_DIM_ROWS
    )
    if _COO_STRUCTURE_OK
    else []
)

# dense_dim selects the trailing dense dimensions; interior values are covered
# at rank 4 and 5, not only 0, 1 and rank-1.
_SMALL_HYBRID_ROWS = [((2, 19, 7), 1)]
_HYBRID_ROWS = (
    (
        [
            ((256,), 0),
            ((1024, 1024), 0),
            ((1024, 1024), 1),
            ((20, 320, 15), 0),
            ((20, 320, 15), 1),
            ((20, 320, 15), 2),
            ((16, 128, 64, 60), 0),
            ((16, 128, 64, 60), 1),
            ((16, 128, 64, 60), 2),
            ((16, 128, 64, 60), 3),
            ((16, 7, 57, 32, 29), 0),
            ((16, 7, 57, 32, 29), 1),
            ((16, 7, 57, 32, 29), 2),
            ((16, 7, 57, 32, 29), 3),
            ((16, 7, 57, 32, 29), 4),
        ]
        + _SMALL_HYBRID_ROWS
    )
    if _COO_STRUCTURE_OK
    else []
)

# Layout targets, including batched compressed conversions at rank 3-5; an
# omitted layout is the schema default (sparse COO). Only the COO targets carry
# the int64 index structure, so a backend without int64 support keeps the
# compressed rows of this table.
_SMALL_LAYOUT_ROWS = [((2, 19, 7), torch.sparse_csr, None)]
_LAYOUT_ROWS = [
    row
    for row in (
        [
            ((), None, None),
            ((1,), None, None),
            ((256,), None, None),
            ((1024, 1024), None, None),
            ((20, 320, 15), None, None),
            ((16, 128, 64, 60), None, None),
            ((16, 7, 57, 32, 29), None, None),
            ((16, 16), torch.sparse_coo, None),
            ((16, 16), torch.sparse_csr, None),
            ((16, 16), torch.sparse_csc, None),
            ((16, 16), torch.sparse_bsr, (2, 2)),
            ((16, 16), torch.sparse_bsc, (2, 2)),
            ((1024, 1024), torch.sparse_csr, None),
            ((1024, 1024), torch.sparse_csc, None),
            ((1024, 1024), torch.sparse_bsr, (16, 16)),
            ((1024, 1024), torch.sparse_bsc, (16, 16)),
            ((20, 320, 15), torch.sparse_csr, None),
            ((20, 320, 15), torch.sparse_csc, None),
            ((20, 320, 15), torch.sparse_bsr, (5, 5)),
            ((20, 320, 15), torch.sparse_bsc, (5, 5)),
            ((16, 128, 64, 60), torch.sparse_csr, None),
            ((16, 128, 64, 60), torch.sparse_bsr, (4, 4)),
            ((16, 7, 57, 32, 29), torch.sparse_csr, None),
        ]
        + _SMALL_LAYOUT_ROWS
    )
    if _COO_STRUCTURE_OK or row[1] not in _COO_TARGETS
]

# Compressed layouts carrying a dense tail: the matrix dimensions sit before the
# tail, so dense_dim moves the batch/matrix/dense split. Unbatched and batched
# rows are both covered.
_COMPRESSED_HYBRID_ROWS = [
    ((8, 8, 4), torch.sparse_csr, None, 1),
    ((2, 8, 8, 4), torch.sparse_csr, None, 1),
    ((16, 16, 4), torch.sparse_csc, None, 1),
    ((8, 8, 4), torch.sparse_bsr, (2, 2), 1),
    ((8, 8, 4, 2), torch.sparse_bsr, (2, 2), 1),
]

# FP8 and complex dtypes on the layouts that natively accept them.
_ALT_LAYOUTS = (
    (torch.sparse_csr, None),
    (torch.sparse_csc, None),
    (torch.sparse_bsr, (4, 4)),
    (torch.sparse_bsc, (4, 4)),
    (torch.strided, None),
)
_SMALL_ALT_ROWS = [
    (dtype, (2, 19, 7), torch.strided, None) for dtype in _FP8_DTYPES + _COMPLEX_DTYPES
]
_ALT_DTYPE_ROWS = (
    [
        (dtype, (16, 16), layout, blocksize)
        for dtype in _FP8_DTYPES + _COMPLEX_DTYPES
        for layout, blocksize in _ALT_LAYOUTS
    ]
    # FP8 cannot use the COO or dense_dim forms (the nonzero kernel has no FP8
    # instantiation), so its shape sweep runs on the strided identity, which is
    # the one form that accepts every shape and FP8 dtype.
    + [
        (dtype, shape, torch.strided, None)
        for dtype in _FP8_DTYPES
        for shape in tu.selected_shapes()
    ]
    + _SMALL_ALT_ROWS
)

# Only FP8 hits the missing nonzero instantiation, so the COO and hybrid forms
# stay positive rows for complex.
_COMPLEX_COO_ROWS = (
    [(dtype, dense_dim) for dtype in _COMPLEX_DTYPES for dense_dim in (None, 1)]
    if _COO_STRUCTURE_OK
    else []
)

# layout=torch.strided is the identity conversion.
_SMALL_STRIDED_ROWS = [
    row
    for row in (
        ((2, 19, 7), "contig", torch.float32),
        ((2, 19, 7), "contig", torch.bfloat16),
    )
    if _device_supports(row[2])
]
_STRIDED_ROWS = [
    row
    for row in (
        ((1024, 1024), "contig", torch.float32),
        ((1024, 1024), "offset", torch.float32),
        ((1024, 1024), "permuted", torch.float32),
        ((16, 128, 64, 60), "contig", torch.float32),
        ((16, 128, 64, 60), "offset", torch.float32),
        ((16, 128, 64, 60), "permuted", torch.bfloat16),
        ((16, 7, 57, 32, 29), "permuted", torch.bfloat16),
        ((20, 320, 15), "offset", torch.bfloat16),
        ((256,), "contig", torch.float32),
        ((), "contig", torch.float64),
        ((16, 16), "expanded", torch.float32),
        ((16, 16), "permuted", torch.complex64),
        ((16, 16), "offset", torch.float8_e4m3fn),
        ((16, 16), "permuted", torch.float8_e5m2),
    )
    if _device_supports(row[2])
] + _SMALL_STRIDED_ROWS

# Omitted optional arguments versus explicit defaults and None.
_SMALL_DEFAULT_ROWS = [((2, 19, 7), {})]
_DEFAULT_ROWS = [
    row
    for row in (
        [
            ((1024, 1024), {}),
            ((1024, 1024), {"layout": torch.strided}),
            ((1024, 1024), {"layout": torch.sparse_coo}),
            ((1024, 1024), {"blocksize": None}),
            ((1024, 1024), {"dense_dim": None}),
            ((20, 320, 15), {}),
            ((20, 320, 15), {"dense_dim": None}),
            ((16, 16), {"dense_dim": None, "blocksize": None}),
            ((16, 128, 64, 60), {}),
        ]
        + _SMALL_DEFAULT_ROWS
    )
    if _COO_STRUCTURE_OK or row[1].get("layout", torch.sparse_coo) not in _COO_TARGETS
]

# Views. The offset rows slice a larger allocation, so they keep contiguous
# strides with a nonzero storage offset, which the compressed targets accept;
# the stride-0 expanded rows and the transposed rows cover the COO/default and
# the strided-identity targets, where arbitrary strides are part of the
# contract.
_VIEW_ROWS = [
    row
    for row in (
        ((16, 16), "offset", torch.sparse_coo, None),
        ((16, 16), "offset", torch.sparse_csr, None),
        ((16, 16), "offset", torch.sparse_csc, None),
        ((16, 16), "expanded", torch.sparse_coo, None),
        ((16, 16), "permuted", None, None),
        ((1024, 1024), "offset", torch.sparse_csr, None),
        ((1024, 1024), "offset", torch.sparse_csc, None),
        ((1024, 1024), "expanded", None, None),
        ((1024, 1024), "permuted", torch.sparse_coo, None),
        ((20, 320, 15), "offset", torch.sparse_csr, None),
        ((16, 128, 64, 60), "offset", torch.sparse_bsr, (4, 4)),
        ((16, 7, 57, 32, 29), "offset", torch.sparse_csr, None),
    )
    if _COO_STRUCTURE_OK or row[2] not in _COO_TARGETS
]

# Sources that are already sparse.
# (shape, source_layout, source_blocksize, target_layout, target_blocksize):
# a block size belongs to the layout that consumes it, so a BSR source keeps its
# own source block while a COO target correctly receives no blocksize. Both the
# source and the target carry int64 index tensors, so this table is gated on the
# int64 capability flag.
_SPARSE_SOURCE_ROWS = (
    [
        ((16, 16), torch.sparse_csr, None, torch.sparse_coo, None),
        ((16, 16), torch.sparse_csr, None, torch.sparse_bsr, (2, 2)),
        ((16, 16), torch.sparse_coo, None, torch.sparse_csr, None),
        ((16, 16), torch.sparse_bsr, (2, 2), torch.sparse_coo, None),
        ((20, 320, 15), torch.sparse_csr, None, torch.sparse_coo, None),
        ((20, 320, 15), torch.sparse_csr, None, torch.sparse_csr, None),
    ]
    if _COO_STRUCTURE_OK
    else []
)

# Stored zeros and duplicate coordinates: the conversion has to keep the stored
# entries (coalescing would drop stored zeros and rewrite nnz), while a
# compressed target has to aggregate duplicates. expected_nnz is filled in only
# where the native stored count is deterministic; otherwise the candidate count
# is compared against the reference alone. The raw stored components include
# int64 indices, so the table is gated on the int64 capability flag.
_SOURCE_EDGE_ROWS = (
    [
        ("duplicate", torch.sparse_csr, None, 3),
        ("duplicate", torch.sparse_coo, None, 5),
        ("duplicate", None, None, 5),
        ("stored_zero", torch.sparse_csr, None, None),
        ("stored_zero", None, None, None),
        ("stored_zero", torch.sparse_bsr, (2, 2), None),
    ]
    if _COO_STRUCTURE_OK
    else []
)

# nnz and extent boundaries. A 1-D compressed input has no native kernel
# (expand(...): the number of sizes provided (1) must be greater or equal), so
# the zero-extent compressed coverage uses 2-D and 3-D shapes. The COO targets
# are gated on the int64 capability flag, the compressed rows are not.
_EDGE_ROWS = [
    row
    for row in (
        ((256,), 0.0, 0, {"sparse_dim": 1}),
        ((256,), 1.0, 256, {"sparse_dim": 1}),
        ((1024, 1024), 0.0, 0, {"sparse_dim": 2}),
        ((1024, 1024), 1.0, 1048576, {"sparse_dim": 2}),
        ((20, 320, 15), 1.0, 6400, {"sparse_dim": 2}),
        ((16, 16), 0.0, 0, {"layout": torch.sparse_csr}),
        ((16, 16), 1.0, 256, {"layout": torch.sparse_csr}),
        ((0,), 0.0, 0, {"sparse_dim": 1}),
        ((0, 5), 0.0, 0, {"sparse_dim": 1}),
        ((0, 5), 0.0, 0, {"layout": torch.sparse_csr}),
        ((5, 0), 0.0, 0, {"layout": torch.sparse_csr}),
        ((0, 16), 0.0, 0, {"layout": torch.sparse_bsr, "blocksize": [2, 2]}),
        ((4, 0, 8), 0.0, 0, {"layout": torch.sparse_csr}),
    )
    if _COO_STRUCTURE_OK or row[3].get("layout", torch.sparse_coo) not in _COO_TARGETS
]

# Backward: pure relocation, plus duplicate-coordinate upstreams that have to
# accumulate (one with exactly representable values, one with floating point
# sums that are not exact). A row is gated on the int64 capability flag when its
# forward result or its upstream gradient carries a sparse index tensor; the
# strided identity row has neither, so it is appended independently.
_STRIDED_BACKWARD_ROWS = [
    ((2, 19, 7), {"layout": torch.strided}, "dense"),
]
_RELOCATION_BACKWARD_ROWS = (
    [
        ((1024, 1024), {"sparse_dim": 2}, "dense"),
        ((20, 320, 15), {"sparse_dim": 2}, "dense"),
        ((1024, 1024), {"sparse_dim": 2}, "coo"),
        ((20, 320, 15), {"sparse_dim": 2}, "coo"),
        ((1024, 1024), {}, "coo"),
        ((16, 16), {"layout": torch.sparse_csr}, "csr"),
        ((16, 16), {"layout": torch.sparse_bsr, "blocksize": [4, 4]}, "bsr"),
    ]
    if _COO_STRUCTURE_OK
    else []
) + _STRIDED_BACKWARD_ROWS
_ACCUMULATION_BACKWARD_ROWS = (
    [
        ((1024, 1024), {"sparse_dim": 2}, "duplicate"),
        ((16, 16), {"sparse_dim": 2}, "fractional"),
    ]
    if _COO_STRUCTURE_OK
    else []
)

# Special values per supported floating dtype; e4m3fn contributes nan only,
# e5m2 contributes nan, inf and mixed.
_SPECIAL_DTYPES = [
    dtype
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    if _device_supports(dtype)
] + list(_FP8_DTYPES)
_SPECIAL_COO_CASES = (
    [
        (dtype, scenario)
        for dtype, scenario in tu.special_value_cases(_SPECIAL_DTYPES)
        if dtype not in _FP8_DTYPES
    ]
    if _COO_STRUCTURE_OK
    else []
)
_SPECIAL_COMPRESSED_CASES = [
    (dtype, scenario, layout, blocksize)
    for dtype, scenario in tu.special_value_cases(_SPECIAL_DTYPES)
    for layout, blocksize in ((torch.sparse_csr, None), (torch.sparse_bsr, (1, 5)))
]

# Invalid arguments: every row raises RuntimeError natively, and AttributeError
# is never an accepted outcome. The FP8 rows come from the capability gate
# above; they are valid parameters that only fail for the missing FP8 nonzero
# kernel.
_NEGATIVE_ROWS = (
    [
        ((16, 16), torch.float32, (5,), {}),
        ((256,), torch.float32, (-1,), {}),
        ((16, 16), torch.float32, (), {"dense_dim": 2}),
        ((16, 16), torch.float32, (), {"blocksize": [2, 2]}),
        (
            (16, 16),
            torch.float32,
            (),
            {"layout": torch.sparse_bsr, "blocksize": [0, 0]},
        ),
        ((8, 8), torch.float32, (), {"layout": torch.sparse_bsr, "blocksize": [3, 3]}),
        (
            (16, 16),
            torch.float32,
            (),
            {"layout": torch.sparse_csr, "blocksize": [2, 2]},
        ),
        ((256,), torch.float32, (), {"layout": torch.sparse_csr}),
        ((), torch.float32, (1,), {}),
    ]
    + [((16, 16), dtype, (2,), {}) for dtype in _FP8_COO_UNSUPPORTED_DTYPES]
    + [((16, 16), dtype, (), {"dense_dim": 1}) for dtype in _FP8_COO_UNSUPPORTED_DTYPES]
)


def _source_tensor(dtype, shape, source_layout, source_blocksize):
    # Already-sparse source, built with the native operator (setup only). The
    # source block size applies only when the source layout is itself blocked.
    dense = _masked_input(dtype, shape, ["-1", "1"], source_layout, source_blocksize)
    if source_layout in (torch.sparse_csr, torch.sparse_csc):
        return torch.ops.aten.to_sparse(dense, layout=source_layout)
    if source_layout in _BLOCK_LAYOUTS:
        blocks = [int(source_blocksize[0]), int(source_blocksize[1])]
        return torch.ops.aten.to_sparse(dense, layout=source_layout, blocksize=blocks)
    return torch.ops.aten.to_sparse(dense)


def _edge_source(kind):
    if kind == "duplicate":
        indices = torch.tensor(
            [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]],
            device=flag_gems.device,
            dtype=torch.int64,
        )
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=flag_gems.device)
        return torch.sparse_coo_tensor(
            indices, values, size=(3, 3), device=flag_gems.device
        )
    indices = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], device=flag_gems.device, dtype=torch.int64
    )
    values = torch.tensor([0.0, 0.0, 7.0], device=flag_gems.device)
    return torch.sparse_coo_tensor(
        indices, values, size=(4, 4), device=flag_gems.device
    )


def _sparse_parts(tensor):
    # Raw stored components in stored order, on the tensor's own device: the
    # actual side of the comparison must stay where the candidate produced it.
    if tensor.layout == torch.sparse_coo:
        return (tensor._indices(), tensor._values())
    return (tensor.crow_indices(), tensor.col_indices(), tensor.values())


def _expected_parts(tensor):
    # Independent reference-device snapshot of the same stored components.
    # tu.to_reference rebuilds the tensor from a clone of its storage, so the
    # snapshot is not aliased by a later mutation of the parent and the raw
    # stored order is preserved without coalescing or densifying the sparse
    # source.
    return tuple(tu.to_reference(part) for part in _sparse_parts(tensor))


def _upstream_gradient(dense, kind):
    # Structural upstream gradient matching the forward result layout, derived
    # from the supplied dense base. The caller passes the same base values to
    # the candidate and the reference through separate to_reference objects, so
    # both graphs receive equal values while neither shares a tensor. Auxiliary
    # tensors are built on the device of the supplied base, so the reference
    # copy stays on the reference device (CPU under --ref cpu).
    if kind == "dense":
        return dense
    if kind == "csr":
        return torch.ops.aten.to_sparse(dense, layout=torch.sparse_csr)
    if kind == "bsr":
        return torch.ops.aten.to_sparse(
            dense, layout=torch.sparse_bsr, blocksize=[4, 4]
        )
    if kind == "bsr_2x2":
        return torch.ops.aten.to_sparse(
            dense, layout=torch.sparse_bsr, blocksize=[2, 2]
        )
    if kind == "coo":
        return torch.ops.aten.to_sparse(dense)
    device = dense.device
    indices = torch.tensor(
        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], device=device, dtype=torch.int64
    )
    if kind == "fractional":
        # Duplicate coordinates whose sums are not exactly representable, so the
        # gradient accumulation is genuinely floating point instead of an exact
        # relocation. The literal is built straight in the payload dtype, so no
        # float64 buffer has to be allocated on the device for the float32 and
        # bfloat16 rows.
        literal = [0.1, 0.7, -0.30000000000000004, 0.001, 0.2]
    else:
        literal = [1.0, 2.0, 3.0, 4.0, 5.0]
    values = torch.tensor(literal, device=device, dtype=dense.dtype)
    return torch.sparse_coo_tensor(indices, values, size=dense.shape, device=device)


_BLOCK_BACKWARD_SHAPE = (4, 4)


def _block_backward_dense(dtype):
    # 2x2 block grid: one all-zero block (dropped by the conversion), one
    # stored block holding an interior zero, and one stored block of ordinary
    # values. A mask derived from the stored element values instead of the block
    # structure would drop the interior zero, which this layout makes visible.
    base = torch.zeros(_BLOCK_BACKWARD_SHAPE, dtype=dtype, device=flag_gems.device)
    base[0, 0] = 1.25
    base[1, 1] = -2.5
    base[2, 3] = 0.75
    return base


def _block_backward_upstream(dtype):
    # Nonuniform upstream, nonzero exactly where the retained block holds its
    # interior zero, so the dense gradient has to keep that position.
    up = torch.zeros(_BLOCK_BACKWARD_SHAPE, dtype=dtype, device=flag_gems.device)
    up[0, 0] = 0.5
    up[0, 1] = -1.75
    up[1, 0] = 3.25
    up[1, 1] = -0.25
    up[2, 2] = 1.5
    up[3, 2] = -0.75
    up[3, 3] = 2.25
    return up


@pytest.mark.to_sparse
@pytest.mark.parametrize("shape,dtype", _GRID_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_to_sparse_sparse_dim_grid(shape, dtype, value_range):
    sparse_dim = min(2, len(shape))
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse(ref_inp, sparse_dim)
    res_out = flag_gems.to_sparse(inp, sparse_dim)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    # The dense source itself must be left untouched.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "shape,sparse_dim", tu.selected_cases(_SPARSE_DIM_ROWS, quick=[])
)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_to_sparse_sparse_dim(shape, sparse_dim, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse(ref_inp, sparse_dim)
    res_out = flag_gems.to_sparse(inp, sparse_dim)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.layout == torch.sparse_coo
    assert res_out.sparse_dim() == sparse_dim
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize("shape,dense_dim", tu.selected_cases(_HYBRID_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_to_sparse_dense_dim(shape, dense_dim, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse(ref_inp, dense_dim=dense_dim)
    res_out = flag_gems.to_sparse(inp, dense_dim=dense_dim)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.layout == torch.sparse_coo
    assert res_out.sparse_dim() == len(shape) - dense_dim
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "shape,layout,blocksize", tu.selected_cases(_LAYOUT_ROWS, quick=[])
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_to_sparse_layouts(shape, layout, blocksize, value_range, dtype):
    inp = _masked_input(dtype, shape, value_range, layout, blocksize)
    ref_inp = tu.to_reference(inp)
    kwargs = _layout_kwargs(layout, blocksize)

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.layout == (torch.sparse_coo if layout is None else layout)
    assert res_out.dtype == dtype
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "shape,layout,blocksize,dense_dim",
    tu.selected_cases(_COMPRESSED_HYBRID_ROWS, quick=[]),
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_to_sparse_compressed_hybrid(
    shape, layout, blocksize, dense_dim, value_range, dtype
):
    inp = _masked_input(dtype, shape, value_range, layout, blocksize, dense_dim)
    ref_inp = tu.to_reference(inp)
    kwargs = _layout_kwargs(layout, blocksize, dense_dim)

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.layout == layout
    assert res_out.dtype == dtype
    assert res_out.dense_dim() == dense_dim
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "dtype,shape,layout,blocksize",
    tu.selected_cases(_ALT_DTYPE_ROWS, quick=[]),
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_to_sparse_fp8_and_complex_layouts(
    dtype, shape, layout, blocksize, value_range
):
    inp = _masked_input(dtype, shape, value_range, layout, blocksize)
    ref_inp = tu.to_reference(inp)
    kwargs = _layout_kwargs(layout, blocksize)

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.dtype == dtype
    assert res_out.layout == layout
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "dtype,dense_dim", tu.selected_cases(_COMPLEX_COO_ROWS, quick=[])
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_to_sparse_complex_coo_forms(dtype, dense_dim, value_range):
    inp = tu.make_input(dtype, (16, 16), value_range)
    ref_inp = tu.to_reference(inp)
    kwargs = {} if dense_dim is None else {"dense_dim": dense_dim}

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.dtype == dtype
    assert res_out.layout == torch.sparse_coo
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "shape,variant,dtype", tu.selected_cases(_STRIDED_ROWS, quick=[])
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_to_sparse_strided_identity(shape, variant, dtype, value_range):
    parent, inp = _view_input(dtype, shape, variant, value_range)
    ref_parent = tu.to_reference(parent)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse(ref_inp, layout=torch.strided)
    res_out = flag_gems.to_sparse(inp, layout=torch.strided)

    tu.assert_result_equal(res_out, ref_out)
    # The strided target is the identity: the result has to stay a view over the
    # same storage, not a re-materialised dense copy.
    assert res_out.device == inp.device
    assert res_out.layout == torch.strided
    assert res_out.data_ptr() == inp.data_ptr()
    assert tuple(res_out.stride()) == tuple(inp.stride())
    assert res_out.storage_offset() == inp.storage_offset()
    # Every element of the parent, including the rows the view does not reach.
    tu.assert_result_equal(parent, ref_parent)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "shape,variant,layout,blocksize", tu.selected_cases(_VIEW_ROWS, quick=[])
)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_to_sparse_view_inputs(shape, variant, layout, blocksize, dtype):
    parent, inp = _view_input(dtype, shape, variant, ["-1", "1"], layout, blocksize)
    ref_parent = tu.to_reference(parent)
    ref_inp = tu.to_reference(inp)
    kwargs = _layout_kwargs(layout, blocksize)

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(parent, ref_parent)


@pytest.mark.to_sparse
@pytest.mark.parametrize("shape,kwargs", tu.selected_cases(_DEFAULT_ROWS, quick=[]))
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_to_sparse_default_arguments(shape, kwargs, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.layout == kwargs.get("layout", torch.sparse_coo)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "shape,source_layout,source_blocksize,layout,blocksize",
    tu.selected_cases(_SPARSE_SOURCE_ROWS, quick=[]),
)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_to_sparse_sparse_source(
    shape, source_layout, source_blocksize, layout, blocksize, dtype
):
    src = _source_tensor(dtype, shape, source_layout, source_blocksize)
    ref_src = tu.to_reference(src)
    before = _expected_parts(src)
    kwargs = _layout_kwargs(layout, blocksize)

    ref_out = torch.ops.aten.to_sparse(ref_src, **kwargs)
    res_out = flag_gems.to_sparse(src, **kwargs)

    if source_layout == layout and source_blocksize == blocksize:
        assert res_out is src
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == src.device
    # The already-sparse source keeps its raw stored data, its layout and its
    # metadata: equal raw components alone would not notice a mutated extent or
    # a flipped coalesced flag.
    for got, want in zip(_sparse_parts(src), before):
        tu.assert_result_equal(got, want)
    assert src.layout == ref_src.layout
    assert tuple(src.shape) == tuple(ref_src.shape)
    assert src._nnz() == ref_src._nnz()
    assert src.sparse_dim() == ref_src.sparse_dim()
    assert src.dense_dim() == ref_src.dense_dim()
    if src.layout == torch.sparse_coo:
        assert src.is_coalesced() == ref_src.is_coalesced()


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "kind,layout,blocksize,expected_nnz",
    tu.selected_cases(_SOURCE_EDGE_ROWS, quick=[]),
)
def test_to_sparse_source_edges(kind, layout, blocksize, expected_nnz):
    src = _edge_source(kind)
    ref_src = tu.to_reference(src)
    before = _expected_parts(src)
    kwargs = _layout_kwargs(layout, blocksize)

    ref_out = torch.ops.aten.to_sparse(ref_src, **kwargs)
    res_out = flag_gems.to_sparse(src, **kwargs)

    if layout in (None, torch.sparse_coo):
        assert res_out is src
    assert res_out.device == src.device
    assert res_out.layout == ref_out.layout
    assert res_out._nnz() == ref_out._nnz()
    if expected_nnz is not None:
        assert res_out._nnz() == expected_nnz
    if res_out.layout == torch.sparse_coo:
        # The stored order of uncoalesced COO is not part of the contract, so
        # compare the materialised values plus the coalesced flag.
        tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())
        assert res_out.is_coalesced() == ref_out.is_coalesced()
    else:
        tu.assert_result_equal(res_out, ref_out)
    for got, want in zip(_sparse_parts(src), before):
        tu.assert_result_equal(got, want)
    assert tuple(src.shape) == tuple(ref_src.shape)
    if src.layout == torch.sparse_coo:
        assert src.is_coalesced() == ref_src.is_coalesced()


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "shape,fill,expected_nnz,kwargs", tu.selected_cases(_EDGE_ROWS, quick=[])
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32] + ([torch.int64] if utils.int64_is_supported else []),
)
def test_to_sparse_extent_boundaries(shape, fill, expected_nnz, kwargs, dtype):
    inp = torch.full(shape, fill, dtype=dtype, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out._nnz() == ref_out._nnz() == expected_nnz
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "shape,kwargs,upstream", tu.selected_cases(_RELOCATION_BACKWARD_ROWS, quick=[])
)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_to_sparse_backward(shape, kwargs, upstream, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp).requires_grad_(True)

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device

    up_base = tu.make_input(dtype, shape, ["-1", "1"])
    res_grad = torch.autograd.grad(
        res_out, inp, grad_outputs=_upstream_gradient(up_base, upstream)
    )[0]
    ref_dgrad_out = torch.autograd.grad(
        ref_out,
        ref_inp,
        grad_outputs=_upstream_gradient(tu.to_reference(up_base), upstream),
    )[0]
    # Relocation only: the gradient is the same rearrangement, so no rounding is
    # involved and the exact comparison applies.
    tu.assert_result_equal(res_grad, ref_dgrad_out)
    assert res_grad.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "shape,kwargs,upstream", tu.selected_cases(_ACCUMULATION_BACKWARD_ROWS, quick=[])
)
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
def test_to_sparse_backward_duplicate_upstream(shape, kwargs, upstream, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp).requires_grad_(True)

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device

    up_base = tu.make_input(dtype, shape, ["-1", "1"])
    res_grad = torch.autograd.grad(
        res_out, inp, grad_outputs=_upstream_gradient(up_base, upstream)
    )[0]
    ref_dgrad_out = torch.autograd.grad(
        ref_out,
        ref_inp,
        grad_outputs=_upstream_gradient(tu.to_reference(up_base), upstream),
    )[0]
    # The duplicated upstream coordinate accumulates instead of relocating, so
    # this row uses the arithmetic comparison while every other gradient row
    # stays exact.
    tu.assert_result_close(res_grad, ref_dgrad_out)
    assert res_grad.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


# The upstream gradient of this row is a BSR tensor, whose index tensors are
# int64, so the row shares the sparse-structure prerequisite of the other
# backward families instead of bypassing it.
_BLOCK_BACKWARD_DTYPES = (
    tu.selected_cases(_BACKWARD_DTYPES, quick=[]) if _COO_STRUCTURE_OK else []
)


@pytest.mark.to_sparse
@pytest.mark.parametrize("dtype", _BLOCK_BACKWARD_DTYPES)
def test_to_sparse_backward_block_interior_zero(dtype):
    inp = _block_backward_dense(dtype).requires_grad_(True)
    ref_inp = tu.to_reference(inp).requires_grad_(True)
    kwargs = {"layout": torch.sparse_bsr, "blocksize": [2, 2]}

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)
    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device

    up = _block_backward_upstream(dtype)
    res_grad = torch.autograd.grad(
        res_out, inp, grad_outputs=_upstream_gradient(up, "bsr_2x2")
    )[0]
    ref_dgrad_out = torch.autograd.grad(
        ref_out,
        ref_inp,
        grad_outputs=_upstream_gradient(tu.to_reference(up), "bsr_2x2"),
    )[0]
    # Densifying the upstream relocation is exact, and the interior zero of the
    # retained block has to keep the upstream value at its position.
    tu.assert_result_equal(res_grad, ref_dgrad_out)
    assert res_grad.device == inp.device
    assert res_grad[0, 1].item() == -1.75
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(_SPECIAL_COO_CASES, quick=[])
)
def test_to_sparse_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_sparse(ref_inp, 1)
    res_out = flag_gems.to_sparse(inp, 1)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize(
    "dtype,scenario,layout,blocksize",
    tu.selected_cases(_SPECIAL_COMPRESSED_CASES, quick=[]),
)
def test_to_sparse_special_values_compressed(dtype, scenario, layout, blocksize):
    inp = tu.make_special_input(dtype, scenario).reshape(1, 5)
    ref_inp = tu.to_reference(inp)
    kwargs = _layout_kwargs(layout, blocksize)

    ref_out = torch.ops.aten.to_sparse(ref_inp, **kwargs)
    res_out = flag_gems.to_sparse(inp, **kwargs)

    tu.assert_result_equal(res_out, ref_out)
    assert res_out.device == inp.device
    assert res_out.dtype == dtype
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.to_sparse
@pytest.mark.parametrize("shape,dtype,args,kwargs", _NEGATIVE_ROWS)
def test_to_sparse_invalid_arguments(shape, dtype, args, kwargs):
    inp = tu.make_input(dtype, shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.to_sparse(inp, *args, **kwargs)
