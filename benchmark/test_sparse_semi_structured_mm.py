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
from flag_gems.utils.device_info import get_device_capability

from . import base, consts

# aten::_sparse_semi_structured_mm CUDA kernel only supports compute capability 8.x.
_ATEN_CC_SUPPORTED = (
    flag_gems.device == "cuda"
    and torch.cuda.is_available()
    and get_device_capability()[0] == 8
)

# Sparse semi-structured MM shapes
SPARSE_SEMI_STRUCTURED_MM_SHAPES = [
    (64, 64),
    (128, 128),
    (256, 128),
    (512, 512),
]

# Within each group of 4 consecutive K elements, reorder columns as
# [0, 2, 1, 3]. The flag_gems bool selector keeps positions {0, 1} when True
# and {2, 3} when False; after this permutation those become {0, 2} and
# {1, 3}, which the CUTLASS sparsifier represents exactly for every
# supported dtype. mat2 rows are permuted identically so the products match.
_GROUP_PERM = torch.tensor([0, 2, 1, 3])


def _to_aten_compressed(dense_mat1, bool_meta, mat2):
    """Convert the benchmark (dense, bool mask, mat2) representation into the
    (packed, swizzled int16 meta, permuted mat2) representation consumed by
    aten::_sparse_semi_structured_mm.

    The bool selector is applied to the dense mat1 first: meta[m, k] = True
    keeps positions 4k and 4k+1, False keeps 4k+2 and 4k+3. Unselected
    positions are zeroed, then the K columns are permuted by _GROUP_PERM so
    the kept pairs become {0, 2}/{1, 3} within each group — patterns the
    CUTLASS sparsifier encodes exactly. Permuting mat2 rows by the same
    permutation keeps the product unchanged.
    """
    from torch.sparse import SparseSemiStructuredTensor

    M, K4 = bool_meta.shape
    keep = torch.where(
        bool_meta.unsqueeze(2),
        torch.tensor(
            [1.0, 1.0, 0.0, 0.0], dtype=dense_mat1.dtype, device=dense_mat1.device
        ),
        torch.tensor(
            [0.0, 0.0, 1.0, 1.0], dtype=dense_mat1.dtype, device=dense_mat1.device
        ),
    ).reshape(M, 4 * K4)
    perm = (torch.arange(K4, device=bool_meta.device) * 4).unsqueeze(
        1
    ) + _GROUP_PERM.to(bool_meta.device)
    perm = perm.reshape(-1)
    masked_mat1 = (dense_mat1 * keep)[:, perm]
    permuted_mat2 = mat2[perm]

    prev_force = SparseSemiStructuredTensor._FORCE_CUTLASS
    SparseSemiStructuredTensor._FORCE_CUTLASS = True
    try:
        sparse = torch.sparse.to_sparse_semi_structured(masked_mat1)
        return sparse.packed, sparse.meta, permuted_mat2
    finally:
        SparseSemiStructuredTensor._FORCE_CUTLASS = prev_force


def _assert_adapters_match():
    """Run both adapters on one sample input per dtype and require matching
    outputs."""
    M, N, K4 = 64, 64, 32
    device = flag_gems.device
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        mat1 = torch.randn(M, 4 * K4, dtype=dtype, device=device)
        mat1_meta = torch.randint(0, 2, (M, K4), dtype=torch.bool, device=device)
        mat2 = torch.randn(4 * K4, N, dtype=dtype, device=device)
        packed, meta, permuted_mat2 = _to_aten_compressed(mat1, mat1_meta, mat2)
        out_aten = _AtenSparseMMAdapter()(
            mat1, mat1_meta, mat2, packed, meta, permuted_mat2
        )
        out_gems = _GemsSparseMMAdapter()(
            mat1, mat1_meta, mat2, packed, meta, permuted_mat2
        )
        torch.testing.assert_close(out_aten, out_gems, rtol=1e-1, atol=1e-1)


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
            yield mat1, mat1_meta, mat2, *_to_aten_compressed(mat1, mat1_meta, mat2)


class _AtenSparseMMAdapter:
    """Adapt the benchmark args to the aten op: the dense (mat1, mat1_meta,
    mat2) prefix follows the flag_gems convention and is consumed by the gems
    side; the trailing pre-compressed (packed, meta, permuted_mat2) triple is
    consumed here."""

    def __call__(self, mat1, mat1_meta, mat2, packed, meta, permuted_mat2):
        return torch.ops.aten._sparse_semi_structured_mm(packed, meta, permuted_mat2)


class _GemsSparseMMAdapter:
    """Consume the shared 6-arg benchmark signature and call the flag_gems
    op with its (mat1, mat1_meta, mat2) prefix."""

    def __call__(self, mat1, mat1_meta, mat2, packed, meta, permuted_mat2):
        return flag_gems._sparse_semi_structured_mm(mat1, mat1_meta, mat2)


@pytest.mark.sparse_semi_structured_mm
@pytest.mark.skipif(
    not _ATEN_CC_SUPPORTED,
    reason="aten::_sparse_semi_structured_mm CUDA kernel only supports compute capability 8.x",
)
def test_sparse_semi_structured_mm():
    _assert_adapters_match()
    bench = SparseSemiStructuredMMBenchmark(
        op_name="sparse_semi_structured_mm",
        torch_op=_AtenSparseMMAdapter(),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(_GemsSparseMMAdapter())
    bench.run()
