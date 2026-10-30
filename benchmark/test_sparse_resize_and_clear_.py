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

import flag_gems

from . import base, consts

# (self shape, target size, sparse_dim, dense_dim). The op replaces COO
# metadata and storage; these paired shapes cover growing, shrinking and a
# hybrid tensor, not pointwise extents.
SPARSE_RESIZE_AND_CLEAR_SHAPES = [
    ((2, 3), (4, 5), 2, 0),
    ((256, 1024), (512, 1024), 2, 0),
    ((16, 4096, 8), (32, 4096, 8), 2, 1),
]


def _torch_sparse_resize_and_clear(indices, values, shape, size, sparse_dim, dense_dim):
    # Construct fresh metadata on both paths so every timed invocation starts
    # from a non-empty tensor; reusing self would only benchmark the empty
    # metadata path after the first call.
    inp = torch.sparse_coo_tensor(indices, values, shape, check_invariants=False)
    return torch.ops.aten.sparse_resize_and_clear_.default(
        inp, size, sparse_dim, dense_dim
    )


def _gems_sparse_resize_and_clear(indices, values, shape, size, sparse_dim, dense_dim):
    inp = torch.sparse_coo_tensor(indices, values, shape, check_invariants=False)
    return flag_gems.sparse_resize_and_clear_(inp, size, sparse_dim, dense_dim)


class SparseResizeAndClearBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_RESIZE_AND_CLEAR_SHAPES
        self.shape_desc = "self_shape, size, sparse_dim, dense_dim"

    def get_input_iter(self, cur_dtype):
        for self_shape, size, sparse_dim, dense_dim in self.shapes:
            dense = torch.randn(self_shape, dtype=cur_dtype, device=self.device)
            inp = dense.to_sparse(sparse_dim)
            yield inp._indices(), inp._values(), self_shape, size, sparse_dim, dense_dim


@pytest.mark.sparse_resize_and_clear_
def test_sparse_resize_and_clear_(monkeypatch):
    # COO clear/resize is host metadata work; use the framework's end-to-end
    # timing mode instead of CUDA graph/event timing, which cannot measure it.
    # Both measured paths include the same fresh metadata construction.
    monkeypatch.setattr(base.Config, "mode", consts.BenchMode.OPERATOR)
    bench = SparseResizeAndClearBenchmark(
        op_name="sparse_resize_and_clear_",
        torch_op=_torch_sparse_resize_and_clear,
        gems_op=_gems_sparse_resize_and_clear,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
