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

# (size, nnz) pairs. The logical extents cover 1-D, 2-D and 3-D COO layouts and
# the entry counts cover sparse (nnz << size) through fully populated payloads.
SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS_SHAPES = [
    ([1024], 256),
    ([4096, 4096], 4096),
    ([128, 512, 512], 8192),
]


def _payload(size, nnz, dtype, device, seed=0):
    """Deterministic COO payload with nnz entries scattered over ``size``."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, nnz, (nnz,), generator=g, dtype=torch.int64)
    indices = torch.stack([base % int(extent) for extent in size])
    values = torch.randn(nnz, dtype=dtype, generator=g).to(device)
    return indices.to(device), values


class SparseCooTensorWithDimsAndTensorsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_COO_TENSOR_WITH_DIMS_AND_TENSORS_SHAPES
        self.shape_desc = "size, nnz"

    def get_input_iter(self, cur_dtype):
        for size, nnz in self.shapes:
            indices, values = _payload(size, nnz, cur_dtype, self.device)
            yield (
                len(size),
                0,
                size,
                indices,
                values,
                {
                    "dtype": cur_dtype,
                    "layout": torch.sparse_coo,
                    "device": self.device,
                },
            )


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test_sparse_coo_tensor_with_dims_and_tensors():
    bench = SparseCooTensorWithDimsAndTensorsBenchmark(
        op_name="sparse_coo_tensor_with_dims_and_tensors",
        torch_op=torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors,
        gems_op=flag_gems.sparse_coo_tensor_with_dims_and_tensors,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
