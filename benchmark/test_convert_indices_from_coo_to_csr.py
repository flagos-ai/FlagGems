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

# ``_convert_indices_from_coo_to_csr`` consumes a rank-1 ascending index vector
# and a row count, so the workload axis is ``(nnz, size)`` rather than a tensor
# shape. ``nnz`` is the executed index count; ``size`` is the pointer-table row
# count (the output holds ``size + 1`` entries).
COO_TO_CSR_CASES = [
    (1, 1),
    (1024, 1024),
    (65536, 4096),
    (1048576, 16384),
    (16777216, 65536),
    (0, 4096),
    (4096, 4096),
]

# The operator only accepts integral index tensors, so the benchmark dtype pool
# is the repository's integer constants rather than the float default.
BENCH_DTYPES = list(consts.INT_DTYPES) + list(consts.EXTRA_INT_DTYPES)


class ConvertIndicesFromCooToCsrBenchmark(base.Benchmark):
    """Benchmark ``_convert_indices_from_coo_to_csr`` over rank-1 workloads.

    Each entry is a ``(nnz, size)`` descriptor: the index vector length and the
    row count handed to the operator.
    """

    def set_shapes(self, shape_file_path=None):
        # The operator-specific (nnz, size) descriptors replace the inherited
        # tensor-shape list, which is not valid geometry for this operator.
        self.shapes = [tuple(entry) for entry in COO_TO_CSR_CASES]

    def get_input_iter(self, cur_dtype):
        info = torch.iinfo(cur_dtype)
        for nnz, size in self.shapes:
            gen = torch.Generator(device="cpu").manual_seed(0)
            bound = min(size, int(info.max) + 1)
            if nnz == 0:
                inp = torch.empty(0, dtype=cur_dtype, device=self.device)
            else:
                inp = torch.randint(0, bound, (nnz,), generator=gen, dtype=torch.int64)
                inp = inp.sort().values.to(cur_dtype).to(self.device)
            yield inp, {"size": size}


@pytest.mark.convert_indices_from_coo_to_csr
def test_convert_indices_from_coo_to_csr():
    bench = ConvertIndicesFromCooToCsrBenchmark(
        op_name="convert_indices_from_coo_to_csr",
        torch_op=torch.ops.aten._convert_indices_from_coo_to_csr,
        dtypes=BENCH_DTYPES,
    )
    bench.run()


class ConvertIndicesFromCooToCsrOutBenchmark(ConvertIndicesFromCooToCsrBenchmark):
    """``.out`` variant: yields the index vector plus a correctly sized buffer."""

    def get_input_iter(self, cur_dtype):
        for inp, kwargs in super().get_input_iter(cur_dtype):
            out = torch.empty(kwargs["size"] + 1, dtype=torch.int64, device=self.device)
            yield inp, {"size": kwargs["size"], "out": out}


@pytest.mark.convert_indices_from_coo_to_csr_out
def test_convert_indices_from_coo_to_csr_out():
    bench = ConvertIndicesFromCooToCsrOutBenchmark(
        op_name="convert_indices_from_coo_to_csr_out",
        torch_op=torch.ops.aten._convert_indices_from_coo_to_csr.out,
        dtypes=BENCH_DTYPES,
    )
    bench.run()
