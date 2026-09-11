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

import pytest
import torch

import flag_gems

from . import base, consts

# aten::_sparse_semi_structured_mm CUDA kernel only supports compute capability 8.x.
_ATEN_CC_SUPPORTED = (
    flag_gems.device == "cuda"
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (8, 0)
) or (
    flag_gems.device == "cuda"
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (8, 6)
)

# Sparse semi-structured MM shapes
SPARSE_SEMI_STRUCTURED_MM_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 128),
    (512, 512),
]


def _to_aten_compressed(dense_mat1, bool_meta):
    """Convert the flag_gems (dense, bool mask) representation into the
    compressed (packed, swizzled int16 meta) representation consumed by
    aten::_sparse_semi_structured_mm.

    flag_gems keeps a dense mat1 of shape (M, 4*K4) plus a per-group bool
    selector, while aten expects the CUTLASS packed mat1 of shape (M, 2*K4)
    and a swizzled int16 meta of shape (M, K4//16). Both describe the same
    underlying 2:4 sparse matrix.
    """
    from torch.sparse import SparseSemiStructuredTensor

    prev_force = SparseSemiStructuredTensor._FORCE_CUTLASS
    SparseSemiStructuredTensor._FORCE_CUTLASS = True
    try:
        sparse = torch.sparse.to_sparse_semi_structured(dense_mat1)
        return sparse.packed, sparse.meta
    finally:
        SparseSemiStructuredTensor._FORCE_CUTLASS = prev_force


class SparseSemiStructuredMMBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_SEMI_STRUCTURED_MM_SHAPES

    def get_input_iter(self, cur_dtype):
        K4 = 32  # K = 4 * K4
        for shape in self.shapes:
            M, N = shape
            mat1 = torch.randn(M, 4 * K4, dtype=cur_dtype, device=self.device)
            mat1_meta = torch.randint(
                0, 2, (M, K4), dtype=torch.bool, device=self.device
            )
            mat2 = torch.randn(4 * K4, N, dtype=cur_dtype, device=self.device)
            yield mat1, mat1_meta, mat2, _to_aten_compressed(mat1, mat1_meta)


class _AtenSparseMMAdapter:
    """Adapt the benchmark args to the aten op: the dense (mat1, mat1_meta,
    mat2) prefix follows the flag_gems convention and is consumed by the gems
    side; the trailing pre-compressed (packed, meta) pair is consumed here."""

    def __call__(self, mat1, mat1_meta, mat2, packed, meta):
        return torch.ops.aten._sparse_semi_structured_mm(packed, meta, mat2)


@pytest.mark.sparse_semi_structured_mm
@pytest.mark.skipif(
    not _ATEN_CC_SUPPORTED,
    reason="aten::_sparse_semi_structured_mm CUDA kernel only supports compute capability 8.x",
)
def test_sparse_semi_structured_mm():
    bench = SparseSemiStructuredMMBenchmark(
        op_name="sparse_semi_structured_mm",
        torch_op=_AtenSparseMMAdapter(),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._sparse_semi_structured_mm)
    bench.run()
