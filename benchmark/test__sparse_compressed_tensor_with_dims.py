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

"""Benchmark for ``aten::_sparse_compressed_tensor_with_dims``.

The operator allocates compressed index and value storage for a requested
structure, so the workload axes are the logical compressed size, the layout
(including the block layouts, which take a derived blocksize) and the number of
stored entries.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_CSR = torch.sparse_csr
_CSC = torch.sparse_csc
_BSR = torch.sparse_bsr
_BSC = torch.sparse_bsc

_LAYOUTS = {"csr": _CSR, "csc": _CSC, "bsr": _BSR, "bsc": _BSC}
_LAYOUT_NAMES = ("csr", "csc", "bsr", "bsc")
_BLOCK_LAYOUTS = ("bsr", "bsc")

# The generic dense core_shapes.yaml has no entry that describes a sparse
# allocation factory, so these compressed sizes are the default; a
# caller-supplied shape file still takes precedence.
_DEFAULT_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

# One stored entry per 64 block-slots of the sparse matrix. Deriving nnz from
# the requested extents keeps the allocation cost growing with the size instead
# of saturating at a fixed cap.
_SLOTS_PER_ENTRY = 64


def _blocksize_for(size):
    nrows, ncols = size[-2], size[-1]
    # A block shape must divide both sparse dims: even pairs use a real 2x2
    # block, anything else falls back to 1x1 so every requested size works.
    if nrows % 2 == 0 and ncols % 2 == 0:
        return (2, 2)
    return (1, 1)


def _nnz_for(size, blocksize):
    nrows, ncols = size[-2], size[-1]
    slots = nrows * ncols
    if blocksize:
        slots //= blocksize[0] * blocksize[1]
    return max(1, slots // _SLOTS_PER_ENTRY) if slots else 0


def _case_fn(shape, dtype):
    """One plan per layout; ``shape`` is the logical compressed size."""
    del dtype
    size = [int(dim) for dim in shape]
    for name in _LAYOUT_NAMES:
        blocksize = list(_blocksize_for(size)) if name in _BLOCK_LAYOUTS else []
        nnz = _nnz_for(size, blocksize)
        yield base.BenchmarkCasePlan(
            shape={"size": size},
            params={
                "layout": str(_LAYOUTS[name]),
                "nnz": nnz,
                "blocksize": blocksize,
                "index_dtype": str(torch.int64),
            },
            builder_args=(name, tuple(size), tuple(blocksize), nnz),
        )


def _build_inputs_fn(plan, dtype, device):
    name, size, blocksize, nnz = plan.builder_args
    # ``nnz`` travels positionally and every remaining schema argument as a
    # keyword, so the reference and the candidate receive the identical call
    # form for every listed case.
    kwargs = {
        "dense_dim": 0,
        "size": list(size),
        "blocksize": list(blocksize),
        "index_dtype": torch.int64,
        "dtype": dtype,
        "layout": _LAYOUTS[name],
        "device": device,
    }
    return nnz, kwargs


class SparseCompressedTensorWithDimsBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Add leading unit axes where the bare shape lacks the two compressed
        # dimensions. Every input element and all original allocation rows remain.
        self.shapes = [
            (1,) * max(0, 2 - len(shape)) + tuple(shape) for shape in self.shapes
        ]
        for shape in _DEFAULT_SHAPES:
            if shape not in self.shapes:
                self.shapes.append(shape)


@pytest.mark.sparse_compressed_tensor_with_dims
def test_sparse_compressed_tensor_with_dims():
    bench = SparseCompressedTensorWithDimsBenchmark(
        op_name="_sparse_compressed_tensor_with_dims",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_compressed_tensor_with_dims,
        gems_op=getattr(flag_gems, "_sparse_compressed_tensor_with_dims", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
