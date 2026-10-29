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

from . import base

# ``indices`` reads the (sparse_dim, nnz) coordinate buffer of a coalesced
# sparse COO tensor, so the workload axis is nnz alone; sparse_dim stays 2.
# This file is shared with the private ``_indices`` benchmark (which carries
# the ``underscore_indices`` marker); this operator's cases use the bare
# ``indices`` marker.
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
    return torch.sparse_coo_tensor(indices, values, size).coalesce()


class IndicesBenchmark(base.Benchmark):
    """Benchmark ``indices`` over coalesced sparse COO coordinate buffers."""

    def set_shapes(self, shape_file_path=None):
        # The operator-specific (nnz, size) descriptors replace the inherited
        # tensor-shape list, which is not valid geometry for sparse inputs.
        self.shapes = [tuple(entry) for entry in SPARSE_INDICES_BENCH_SHAPES]

    def get_input_iter(self, cur_dtype):
        for nnz, size in self.shapes:
            inp = _make_coo(nnz, size, self.device)
            yield (inp,)


@pytest.mark.indices
def test_indices():
    bench = IndicesBenchmark(
        op_name="indices",
        torch_op=torch.ops.aten.indices,
        # The result is an aliasing metadata view; the values dtype is fixed
        # by the layout, so the dtype pool only drives input construction.
        dtypes=[torch.float32],
    )
    bench.run()
