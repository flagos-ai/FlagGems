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

# (sparse shape, nnz) workload descriptors, mirroring the KernelGen acceptance
# benchmark. Coalescing work scales with nnz; nnz > numel(shape) guarantees
# duplicate coordinates and therefore real sort/merge work.
_COALESCE_SHAPES = [
    ((1024, 1024), 65536),
    ((1024, 1024), 262144),
    ((1024, 1024), 1048576),
    ((4096, 4096), 1048576),
    ((2048, 2048), 2097152),
    ((256, 256, 256), 1048576),
]


def _make_uncoalesced_coo(sparse_shape, nnz, dtype, device):
    # Draw coordinates with replacement so the input is uncoalesced (the CUDA
    # reference asserts on that) and coalescing has to sort/merge.
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
            for dim in sparse_shape
        ]
    )
    values = torch.randn(nnz, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, sparse_shape, device=device)


class CoalesceBenchmark(base.Benchmark):
    """``aten::coalesce`` over duplicate-bearing sparse COO tensors.

    ``coalesce`` is a sparse op with no meaningful dense shapes, so the
    benchmark sweeps the KernelGen (sparse_shape, nnz) workload descriptors.
    """

    def set_shapes(self, shape_file_path=None):
        # (sparse_shape, nnz) pairs; nnz > prod(shape) guarantees duplicates.
        self.shapes = _COALESCE_SHAPES

    def get_input_iter(self, cur_dtype):
        for sparse_shape, nnz in self.shapes:
            inp = _make_uncoalesced_coo(sparse_shape, nnz, cur_dtype, self.device)
            yield inp,


@pytest.mark.coalesce
def test_coalesce():
    bench = CoalesceBenchmark(
        op_name="coalesce",
        torch_op=torch.ops.aten.coalesce,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.underscore_coalesce
def test__coalesce():
    bench = CoalesceBenchmark(
        op_name="_coalesce",
        torch_op=torch.ops.aten._coalesce,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class CoalesceOutBenchmark(CoalesceBenchmark):
    """``.out`` variant: yields the sparse input plus an out buffer."""

    def get_input_iter(self, cur_dtype):
        for sparse_shape, nnz in self.shapes:
            inp = _make_uncoalesced_coo(sparse_shape, nnz, cur_dtype, self.device)
            out = torch.sparse_coo_tensor(
                torch.zeros(
                    (len(sparse_shape), 1), dtype=torch.long, device=self.device
                ),
                torch.zeros((1,), dtype=cur_dtype, device=self.device),
                sparse_shape,
                is_coalesced=True,
            )
            yield inp, {"out": out}


@pytest.mark.underscore_coalesce_out
def test__coalesce_out():
    bench = CoalesceOutBenchmark(
        op_name="_coalesce_out",
        torch_op=torch.ops.aten._coalesce.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
