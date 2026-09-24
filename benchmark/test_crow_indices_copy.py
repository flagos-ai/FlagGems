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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts

# (layout, size, nnz, blocks). crow_indices_copy materializes the
# batch_dims + (n_compressed + 1,) compressed-row pointer array of a sparse CSR
# or BSR tensor as a fresh contiguous int64 copy. It is a metadata accessor
# whose cost is proportional to the number of compressed rows (the size of the
# returned array) and independent of the stored values, so the benchmark sweeps
# a spread of row extents, batch dims and block sizes. Only nnz entries are
# stored on the device, keeping the values allocation small relative to the
# logical size.
_CROW = [
    ("csr", (1024, 1024), 65536, None),
    ("csr", (4096, 4096), 1048576, None),
    ("csr", (65536, 1024), 1048576, None),
    ("csr", (262144, 64), 1048576, None),
    ("csr_batch", (8, 4096, 4096), 131072, None),
    ("bsr", (4096, 4096), 262144, (8, 8)),
    ("bsr", (8192, 8192), 65536, (16, 16)),
    ("bsr_batch", (4, 4096, 4096), 65536, (8, 8)),
]


def _random_values(shape, dtype, gen):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, generator=gen)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, generator=gen)
    return torch.randint(-5, 6, shape, dtype=dtype, generator=gen)


def _make_crow(n_compressed, nnz):
    counts = torch.full((n_compressed,), nnz // n_compressed, dtype=torch.long)
    counts[: nnz % n_compressed] += 1
    return torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])


def _make_input(layout, size, nnz, blocks, dtype, device):
    gen = torch.Generator("cpu").manual_seed(0)
    if layout == "csr":
        n_rows, n_cols = size
        assert 0 <= nnz <= n_rows * n_cols
        crow = _make_crow(n_rows, nnz)
        col = torch.arange(nnz) - torch.repeat_interleave(crow[:-1], crow.diff())
        values = _random_values((nnz,), dtype, gen)
        return torch.sparse_csr_tensor(crow, col, values, size=size, device=device)
    if layout == "csr_batch":
        batch, n_rows, n_cols = size
        assert 0 <= nnz <= n_rows * n_cols
        crows, cols, values = [], [], []
        for _ in range(batch):
            crows.append(_make_crow(n_rows, nnz))
            cols.append(
                torch.arange(nnz)
                - torch.repeat_interleave(crows[-1][:-1], crows[-1].diff())
            )
            values.append(_random_values((nnz,), dtype, gen))
        return torch.sparse_csr_tensor(
            torch.stack(crows),
            torch.stack(cols),
            torch.stack(values),
            size=size,
            device=device,
        )
    if layout == "bsr":
        n_rows, n_cols = size
        br, bc = blocks
        assert n_rows % br == n_cols % bc == 0
        n_row_blocks = n_rows // br
        n_col_blocks = n_cols // bc
        assert 0 <= nnz <= n_row_blocks * n_col_blocks
        crow = _make_crow(n_row_blocks, nnz)
        col = torch.arange(nnz) - torch.repeat_interleave(crow[:-1], crow.diff())
        values = _random_values((nnz, br, bc), dtype, gen)
        return torch.sparse_bsr_tensor(crow, col, values, size=size, device=device)
    if layout == "bsr_batch":
        batch, n_rows, n_cols = size
        br, bc = blocks
        assert n_rows % br == n_cols % bc == 0
        n_row_blocks = n_rows // br
        n_col_blocks = n_cols // bc
        assert 0 <= nnz <= n_row_blocks * n_col_blocks
        crows, cols, values = [], [], []
        for _ in range(batch):
            crows.append(_make_crow(n_row_blocks, nnz))
            cols.append(
                torch.arange(nnz)
                - torch.repeat_interleave(crows[-1][:-1], crows[-1].diff())
            )
            values.append(_random_values((nnz, br, bc), dtype, gen))
        return torch.sparse_bsr_tensor(
            torch.stack(crows),
            torch.stack(cols),
            torch.stack(values),
            size=size,
            device=device,
        )
    raise ValueError(f"unknown layout {layout}")


def _case_fn(shape, dtype):
    del dtype
    layout, size, nnz, blocks = shape
    yield base.BenchmarkCasePlan(
        shape={"input": size},
        params={"nnz": nnz, "layout": layout},
        builder_args=(layout, size, nnz, blocks),
    )


def _build_inputs_fn(plan, dtype, device):
    layout, size, nnz, blocks = plan.builder_args
    inp = _make_input(layout, size, nnz, blocks, dtype, device)
    return inp, {}


class CrowIndicesCopyBenchmark(OperatorBenchmark):
    # crow_indices_copy is a sparse metadata accessor; there are no meaningful
    # dense shapes in core_shapes.yaml, so benchmark dedicated
    # (layout, size, nnz, blocks) cases instead.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_CROW)


@pytest.mark.crow_indices_copy
def test_crow_indices_copy():
    bench = CrowIndicesCopyBenchmark(
        op_name="crow_indices_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.crow_indices_copy,
        gems_op=getattr(flag_gems, "crow_indices_copy", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
