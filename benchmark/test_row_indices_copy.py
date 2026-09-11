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

from . import base, consts

# Sparse CSC matrix configurations (rows, cols, nnz). row_indices_copy copies
# the row_indices buffer of a column-compressed sparse tensor, so the benchmark
# exercises a range of nnz counts (which is the size of the copied buffer).
ROW_INDICES_COPY_SHAPES = [
    (256, 256, 1024),
    (1024, 1024, 4096),
    (1024, 1024, 16384),
    (4096, 4096, 16384),
    (4096, 4096, 65536),
]

# Index dtypes supported for sparse buffers (the dtype of the copied output).
INDEX_DTYPES = [torch.int32, torch.int64]


def _make_csc(shape, index_dtype, device):
    rows, cols, nnz = shape
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1024)
    per_col = torch.randint(0, max(nnz, 1) * 2, (cols,), generator=gen).tolist()
    if sum(per_col) == 0:
        per_col = [1] * min(cols, nnz) if cols else []
    scale = nnz / max(sum(per_col), 1)
    per_col = [max(int(round(c * scale)), 0) for c in per_col]
    if per_col:
        per_col[-1] = max(per_col[-1] + (nnz - sum(per_col)), 0)
    else:
        per_col = [nnz]
    ccol = torch.zeros(cols + 1, dtype=index_dtype, device="cpu")
    acc = 0
    for i in range(cols):
        acc += per_col[i]
        ccol[i + 1] = acc
    ccol = ccol.to(device).to(index_dtype)
    row_list = []
    for i in range(cols):
        n = int(ccol[i + 1].item() - ccol[i].item())
        if n > 0:
            rs = torch.randperm(rows, generator=gen)[:n]
            rs, _ = torch.sort(rs)
            row_list.append(rs)
    if row_list:
        row = torch.cat(row_list).to(index_dtype).to(device)
    else:
        row = torch.empty(0, dtype=index_dtype, device=device)
    vals = torch.randn(row.numel(), dtype=torch.float32, device=device)
    return torch.sparse_csc_tensor(ccol, row, vals, size=(rows, cols))


class RowIndicesCopyBenchmark(base.Benchmark):
    """Benchmark for row_indices_copy on column-compressed sparse tensors."""

    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = ROW_INDICES_COPY_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            csc = _make_csc(shape, cur_dtype, self.device)
            yield (csc,)


@pytest.mark.row_indices_copy
@pytest.mark.parametrize("in_dtype", INDEX_DTYPES)
def test_row_indices_copy(in_dtype):
    bench = RowIndicesCopyBenchmark(
        op_name="row_indices_copy",
        torch_op=torch.row_indices_copy,
        dtypes=[in_dtype],
    )
    bench.run()


@pytest.mark.row_indices_copy_out
@pytest.mark.parametrize("in_dtype", INDEX_DTYPES)
def test_row_indices_copy_out(in_dtype):
    bench = RowIndicesCopyBenchmark(
        op_name="row_indices_copy_out",
        torch_op=torch.row_indices_copy,
        dtypes=[in_dtype],
    )
    # Provide a pre-allocated out tensor sized to the buffer for each input.
    base_get_input_iter = bench.get_input_iter

    def get_input_iter_with_out(cur_dtype):
        for (csc,) in base_get_input_iter(cur_dtype):
            n = csc.row_indices().numel()
            out = torch.empty(n, dtype=cur_dtype, device=bench.device)
            # The dict is unpacked into kwargs by the benchmark harness.
            yield (csc, {"out": out})

    bench.get_input_iter = get_input_iter_with_out
    bench.run()
