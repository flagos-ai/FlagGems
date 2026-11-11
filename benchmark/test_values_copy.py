# Copyright 2026, The FlagOS Contributors.
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

from . import base, consts

_LAYOUTS = {
    "csr": torch.sparse_csr_tensor,
    "csc": torch.sparse_csc_tensor,
    "bsr": torch.sparse_bsr_tensor,
    "bsc": torch.sparse_bsc_tensor,
}
_BLOCK_LAYOUTS = ("bsr", "bsc")

# (layout, size, nnz, blocks, nbatch), mirroring the KernelGen acceptance
# benchmark. values_copy materializes the stored values of a compressed sparse
# tensor, so its cost follows the stored values - nnz times the BSR/BSC block
# or the dense trailing dims - and not the logical matrix size. The cases sweep
# nnz magnitudes, dense trailing dims, block layouts, one and two batch axes,
# and a zero-extent storage; only nnz values per batch are allocated.
_CASES = [
    ("csr", (1024, 1024), 65536, None, 0),
    ("csr", (4096, 4096), 1048576, None, 0),
    ("csr", (1024, 1024, 64), 65536, None, 0),
    ("csr", (8, 4096, 4096), 131072, None, 1),
    ("csc", (4096, 4096), 1048576, None, 0),
    ("bsr", (4096, 4096), 262144, (8, 8), 0),
    ("bsr", (4, 4096, 4096), 65536, (8, 8), 1),
    ("bsc", (4096, 4096), 262144, (8, 8), 0),
    ("csr", (4, 2, 4096, 4096), 131072, None, 2),
    ("csr", (0, 1024), 0, None, 0),
]

_INDEX_DTYPE = torch.int64


def _geometry(layout, size, nnz, blocks, nbatch):
    """Validate one case row and return its index/values metadata."""
    size = tuple(size)
    batch = size[:nbatch]
    m, n = size[nbatch], size[nbatch + 1]
    dense = size[nbatch + 2 :]
    if layout in _BLOCK_LAYOUTS:
        blocks = tuple(blocks)
        block_rows, block_cols = blocks
        groups = m // block_rows if layout == "bsr" else n // block_cols
    elif layout == "csc":
        groups = n
    else:
        groups = m
    values_shape = batch + (nnz,) + tuple(blocks or ()) + dense
    return {
        "batch": list(batch),
        "dense": list(dense),
        "groups": groups,
        "blocks": list(blocks) if blocks else None,
        "values_shape": list(values_shape),
    }


def _stored_values(shape, dtype, device):
    """Generate stored values on the target device."""
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device, dtype=torch.bool)
    if dtype == torch.uint8:
        return torch.randint(0, 6, shape, device=device, dtype=torch.uint8)
    return torch.randint(-5, 6, shape, device=device, dtype=dtype)


def _make_input(layout, size, nnz, blocks, nbatch, dtype, device):
    meta = _geometry(layout, size, nnz, blocks, nbatch)
    groups = meta["groups"]
    if groups == 0:
        # Zero-extent storage: one empty compressed-offset vector, no plain index.
        compressed = torch.zeros(1, dtype=_INDEX_DTYPE, device=device)
        plain = torch.empty(0, dtype=_INDEX_DTYPE, device=device)
    else:
        counts = torch.full((groups,), nnz // groups, dtype=_INDEX_DTYPE, device=device)
        counts[: nnz % groups] += 1
        compressed = torch.cat(
            [
                torch.zeros(1, dtype=_INDEX_DTYPE, device=device),
                counts.cumsum(0, dtype=_INDEX_DTYPE),
            ]
        )
        plain = torch.arange(nnz, device=device, dtype=_INDEX_DTYPE) - (
            torch.repeat_interleave(compressed[:-1], counts)
        )
    if meta["batch"]:
        compressed = compressed.expand(*meta["batch"], -1).contiguous()
        plain = plain.expand(*meta["batch"], -1).contiguous()
    values = _stored_values(tuple(meta["values_shape"]), dtype, device)
    return _LAYOUTS[layout](compressed, plain, values, size=tuple(size), device=device)


class ValuesCopyBenchmark(base.Benchmark):
    """``aten::values_copy`` over compressed-sparse values tensors.

    The op copies the stored values, so the benchmark sweeps the KernelGen
    (layout, size, nnz, blocks, nbatch) descriptors.
    """

    def set_shapes(self, shape_file_path=None):
        # (layout, size, nnz, blocks, nbatch) tuples; nnz drives the workload.
        self.shapes = _CASES

    def get_input_iter(self, cur_dtype):
        for layout, size, nnz, blocks, nbatch in self.shapes:
            inp = _make_input(layout, size, nnz, blocks, nbatch, cur_dtype, self.device)
            yield inp,


@pytest.mark.values_copy
def test_values_copy():
    bench = ValuesCopyBenchmark(
        op_name="values_copy",
        torch_op=torch.ops.aten.values_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class ValuesCopyOutBenchmark(ValuesCopyBenchmark):
    """``.out`` variant: yields the sparse input plus an out buffer."""

    def get_input_iter(self, cur_dtype):
        for layout, size, nnz, blocks, nbatch in self.shapes:
            inp = _make_input(layout, size, nnz, blocks, nbatch, cur_dtype, self.device)
            out = torch.empty_like(inp.values())
            yield inp, {"out": out}


@pytest.mark.values_copy_out
def test_values_copy_out():
    bench = ValuesCopyOutBenchmark(
        op_name="values_copy_out",
        torch_op=torch.ops.aten.values_copy.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# Sparse COO shapes for the _values_copy benchmark: (nnz, size, dense_tail).
# The op copies the (nnz,) + dense_tail values tensor; nnz drives the workload
# and sparse_dim stays 2.
SPARSE_VALUES_BENCH_SHAPES = [
    (8, (4, 4), ()),
    (1024, (32, 32), ()),
    (4096, (64, 64, 2), (2,)),
    (16384, (128, 128), ()),
    (65536, (256, 256, 2), (2,)),
]


def _make_coo(nnz, size, dense_tail, dtype, device):
    numel = 1
    for s in size:
        numel *= s
    dense_tail = tuple(dense_tail)
    nnd = len(size) - len(dense_tail)
    flat = torch.randperm(numel)[:nnz]
    indices = torch.empty((nnd, nnz), dtype=torch.int64)
    work = flat.clone()
    for d in reversed(range(nnd)):
        indices[d] = work % size[d]
        work = work // size[d]
    perm = torch.arange(nnz)
    for d in reversed(range(nnd)):
        perm = perm[torch.argsort(indices[d][perm], stable=True)]
    indices = indices[:, perm].contiguous().to(device)
    values = torch.randn((nnz,) + tuple(dense_tail), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, size)


class UnderscoreValuesCopyBenchmark(base.Benchmark):
    """``aten::_values_copy`` over sparse COO values tensors.

    The op copies the (nnz,) + dense_tail values tensor, so nnz drives the
    workload.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_VALUES_BENCH_SHAPES

    def get_input_iter(self, cur_dtype):
        for nnz, size, dense_tail in self.shapes:
            inp = _make_coo(nnz, size, dense_tail, cur_dtype, self.device)
            yield inp, {}


def _values_copy_ref(self):
    # The native op has no CUDA sparse-COO kernel; the equivalent baseline is
    # the plain values accessor materialized with clone.
    return self._values().clone()


def _values_copy_out_ref(self, out):
    # Same rationale as _values_copy_ref: materialize the stored values into the
    # provided out buffer.
    out.copy_(self._values())
    return out


@pytest.mark.underscore_values_copy
def test__values_copy():
    bench = UnderscoreValuesCopyBenchmark(
        op_name="_values_copy",
        torch_op=_values_copy_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class UnderscoreValuesCopyOutBenchmark(UnderscoreValuesCopyBenchmark):
    """``.out`` variant: yields the sparse input plus an out buffer."""

    def get_input_iter(self, cur_dtype):
        for nnz, size, dense_tail in self.shapes:
            inp = _make_coo(nnz, size, dense_tail, cur_dtype, self.device)
            out = torch.empty_like(inp._values())
            yield inp, {"out": out}


@pytest.mark.underscore_values_copy_out
def test__values_copy_out():
    bench = UnderscoreValuesCopyOutBenchmark(
        op_name="_values_copy_out",
        torch_op=_values_copy_out_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
