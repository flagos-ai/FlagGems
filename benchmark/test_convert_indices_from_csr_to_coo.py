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

# ``_convert_indices_from_csr_to_coo`` decodes one COO entry per stored
# column index, so the workload axis is the leading batch shape plus the stored
# entry count ``nnz``; the row count follows from the block shape. Column
# shapes are ``lead + (nnz,)`` and the row-pointer shape is
# ``lead + (rows + 1,)``.
CSR_TO_COO_CASES = [
    # Small descriptors first: every index dtype can address them, so each
    # dtype in the pool contributes at least one measured case.
    ((), 8, 8),
    ((2, 2), 4, 4),
    ((), 1024, 65536),
    ((8,), 512, 8192),
    ((1024,), 256, 4096),
    ((64, 16), 256, 4096),
    ((16, 8, 8), 128, 2048),
]

# Only integral index tensors are accepted, so the benchmark dtype pool is the
# repository's integer constants rather than the float default.
BENCH_DTYPES = list(consts.INT_DTYPES) + list(consts.EXTRA_INT_DTYPES)


def _make_csr(lead, rows, nnz, dtype, device):
    """Valid CSR operands: monotonic pointers ending at ``nnz`` per block."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    cuts = torch.sort(
        torch.randint(0, nnz + 1, lead + (rows,), generator=gen, dtype=torch.int64),
        dim=-1,
    ).values
    crow = torch.cat([torch.zeros(lead + (1,), dtype=torch.int64), cuts], dim=-1)
    crow[..., -1] = nnz
    col = torch.randint(0, rows, lead + (nnz,), generator=gen, dtype=torch.int64)
    return crow.to(dtype).to(device), col.to(dtype).to(device)


class CsrToCooBenchmark(base.Benchmark):
    """Benchmark ``_convert_indices_from_csr_to_coo`` over CSR workloads."""

    def set_shapes(self, shape_file_path=None):
        # The operator-specific (lead, rows, nnz) descriptors replace the
        # inherited tensor-shape list, which is not valid geometry here.
        self.shapes = [tuple(entry) for entry in CSR_TO_COO_CASES]

    def get_input_iter(self, cur_dtype):
        # The row pointers must be representable in the index dtype: a narrow
        # dtype cannot address a larger entry count, so those descriptors are
        # skipped instead of being built from wrapped pointer values.
        max_index = int(torch.iinfo(cur_dtype).max)
        for lead, rows, nnz in self.shapes:
            if nnz > max_index or rows > max_index:
                continue
            crow, col = _make_csr(lead, rows, nnz, cur_dtype, self.device)
            yield crow, col


@pytest.mark.convert_indices_from_csr_to_coo
def test_convert_indices_from_csr_to_coo():
    bench = CsrToCooBenchmark(
        op_name="convert_indices_from_csr_to_coo",
        torch_op=torch.ops.aten._convert_indices_from_csr_to_coo,
        dtypes=BENCH_DTYPES,
    )
    bench.run()


class CsrToCooOutBenchmark(CsrToCooBenchmark):
    """``.out`` variant: yields the operands plus a correctly sized buffer."""

    def get_input_iter(self, cur_dtype):
        for crow, col in super().get_input_iter(cur_dtype):
            nd = crow.dim() - 1
            total = crow.numel() // (crow.shape[-1]) * col.shape[-1]
            out = torch.empty((nd + 2, total), dtype=torch.int64, device=self.device)
            yield crow, col, {"out": out}


@pytest.mark.convert_indices_from_csr_to_coo_out
def test_convert_indices_from_csr_to_coo_out():
    bench = CsrToCooOutBenchmark(
        op_name="convert_indices_from_csr_to_coo_out",
        torch_op=torch.ops.aten._convert_indices_from_csr_to_coo.out,
        dtypes=BENCH_DTYPES,
    )
    bench.run()
