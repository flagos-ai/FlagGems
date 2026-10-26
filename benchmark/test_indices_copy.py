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

# Sparse COO shapes for the indices_copy benchmark: (nnz, size). The op is a
# flat copy of the (sparse_dim, nnz) index buffer, so nnz drives the workload;
# sparse_dim stays 2 so the buffer stays coalesced.
SPARSE_INDICES_BENCH_SHAPES = [
    (8, (4, 4)),
    (1024, (32, 32)),
    (4096, (64, 64)),
    (16384, (128, 128)),
    (65536, (256, 256)),
]


def _make_coo(nnz, size, device):
    numel = 1
    for s in size:
        numel *= s
    nnd = len(size)
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
    values = torch.randn(nnz, device=device)
    return torch.sparse_coo_tensor(indices, values, size)


class IndicesCopyBenchmark(base.Benchmark):
    """``aten::indices_copy`` over sparse COO index buffers.

    The op copies the (sparse_dim, nnz) index tensor, so the workload axis is
    nnz alone.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_INDICES_BENCH_SHAPES

    def get_input_iter(self, cur_dtype):
        for nnz, size in self.shapes:
            inp = _make_coo(nnz, size, self.device)
            yield inp, {}


def indices_copy_ref(self):
    # The native op has no CUDA sparse-COO kernel; the equivalent baseline is
    # the plain indices accessor materialized with clone.
    return self._indices().clone()


def indices_copy_out_ref(self, out):
    # The native op has no CUDA sparse-COO kernel; the equivalent baseline is
    # materializing the indices accessor into the provided out buffer.
    out.copy_(self._indices())
    return out


@pytest.mark.indices_copy
def test_indices_copy():
    bench = IndicesCopyBenchmark(
        op_name="indices_copy",
        torch_op=indices_copy_ref,
        # Index copies are dtype-agnostic (int64 buffers); float32 only
        # drives the workload generator.
        dtypes=[torch.float32],
    )
    bench.run()


class IndicesCopyOutBenchmark(IndicesCopyBenchmark):
    """``.out`` variant: yields the sparse input plus an out buffer."""

    def get_input_iter(self, cur_dtype):
        for nnz, size in self.shapes:
            inp = _make_coo(nnz, size, self.device)
            out = torch.empty_like(inp._indices())
            yield inp, {"out": out}


@pytest.mark.indices_copy_out
def test_indices_copy_out():
    bench = IndicesCopyOutBenchmark(
        op_name="indices_copy_out",
        torch_op=indices_copy_out_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
