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

import statistics
import time

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import base, consts

# Match the historical workload, but measure both construction paths afresh.
SPARSE_COO_TENSOR_SHAPES = [
    ((4096, 4096), 10000, ()),
    ((4096, 4096), 100000, ()),
    ((1024, 8192), 50000, ()),
    ((8192, 1024), 50000, ()),
    ((2048, 2048, 64), 20000, (64,)),
]


class SparseCooTensorBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_COO_TENSOR_SHAPES
        self.shape_desc = "logical shape, nnz, dense shape; synchronized wall time"

    def get_input_iter(self, cur_dtype):
        for shape, nnz, dense_shape in self.shapes:
            sparse_dim = len(shape) - len(dense_shape)
            indices = torch.stack(
                [
                    torch.randint(shape[d], (nnz,), device=self.device)
                    for d in range(sparse_dim)
                ]
            )
            values = torch.randn(
                (nnz, *dense_shape), dtype=cur_dtype, device=self.device
            )
            if self.infer_size:
                yield indices, values
            else:
                yield indices, values, shape

    def get_latency(self, op, *args, **kwargs):
        # CUDA events/graphs do not measure host sparse-container construction.
        # Include the inferred-size device-to-host synchronization as well.
        for _ in range(10):
            op(*args, **kwargs)
        samples = []
        for _ in range(5):
            torch_device_fn.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                op(*args, **kwargs)
            torch_device_fn.synchronize()
            samples.append((time.perf_counter() - start) * 1000 / 100)
        return statistics.median(samples)


@pytest.mark.sparse_coo_tensor
@pytest.mark.parametrize("infer_size", [False, True], ids=["explicit", "inferred"])
def test_sparse_coo_tensor(infer_size, monkeypatch):
    monkeypatch.setattr(base.Config, "mode", consts.BenchMode.OPERATOR)
    bench = SparseCooTensorBenchmark(
        op_name="sparse_coo_tensor",
        torch_op=torch.sparse_coo_tensor,
        gems_op=flag_gems.sparse_coo_tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.infer_size = infer_size
    bench.run()
