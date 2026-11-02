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

# Descriptor shapes chosen to cover 1-D, 2-D and 3-D COO layouts; the operator
# allocates an empty descriptor, so the logical extents are what matter.
SPARSE_COO_TENSOR_WITH_DIMS_SHAPES = [
    [1024],
    [1024, 1024],
    [64, 512, 512],
]


class SparseCooTensorWithDimsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_COO_TENSOR_WITH_DIMS_SHAPES
        self.shape_desc = "size (sparse_dim = len(size))"

    def get_input_iter(self, cur_dtype):
        for size in self.shapes:
            yield (
                len(size),
                0,
                size,
                {"dtype": cur_dtype, "layout": torch.sparse_coo},
            )


@pytest.mark.sparse_coo_tensor_with_dims
def test_sparse_coo_tensor_with_dims():
    bench = SparseCooTensorWithDimsBenchmark(
        op_name="sparse_coo_tensor_with_dims",
        torch_op=torch.ops.aten._sparse_coo_tensor_with_dims,
        gems_op=flag_gems.sparse_coo_tensor_with_dims,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
