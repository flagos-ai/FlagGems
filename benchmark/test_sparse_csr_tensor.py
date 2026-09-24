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

# aten::sparse_csr_tensor.crow_col_value_size(Tensor crow_indices,
#     Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None,
#     ...) -> Tensor constructs a sparse CSR tensor from its raw components: the
# (rows, cols) trailing dims of `size` are spanned by the row-pointer and
# column arrays, with `values` holding nnz entries (or (batch, nnz) for
# batched tensors). The measured work is the layout construction from the three
# component tensors, so the benchmark feeds the components directly (not a
# pre-built sparse tensor) and both the reference and the candidate receive the
# exact same call.
#
# The factory has no elementwise/unary/reduction analogue, so no public
# Benchmark family applies; the two-phase GenericBenchmark (case_fn +
# build_inputs_fn) is used instead. `gems_op` is resolved through
# `getattr` because the candidate is injected by the KernelGen harness via
# `--override` and is not necessarily present as a public
# `flag_gems.sparse_csr_tensor` attribute at import time.
#
# Each benchmark case is (tensor_shape, nnz). The nnz is distributed across the
# rows with a deterministic pattern, so the crow/col arrays and the values
# allocation all scale with nnz while the logical matrix spans the full
# (rows, cols) extent.
_BENCH_SHAPES = [
    ((512, 512), 65536),
    ((1024, 1024), 262144),
    ((2048, 2048), 1048576),
    ((4096, 4096), 2097152),
    ((64, 512, 512), 65536),
    ((16, 1024, 1024), 262144),
]


def _make_csr_inputs(shape, nnz, dtype, device, seed=0):
    # Deterministic CPU-side generation of a valid (crow_indices, col_indices,
    # values) triple for the given logical shape and nnz, moved to the benchmark
    # device. The nnz is spread over the rows by an affine split so every row
    # gets either floor or ceil of nnz / n_rows entries; for batched shapes the
    # batch dims are prepended to the index arrays (every batch shares the same
    # row-block structure, which is a valid batched CSR layout).
    gen = torch.Generator("cpu").manual_seed(seed)
    batch = shape[:-2]
    rows, cols = shape[-2], shape[-1]
    assert 0 <= nnz <= rows * cols
    counts = torch.full((rows,), nnz // rows, dtype=torch.long)
    counts[: nnz % rows] += 1
    crow = torch.zeros(rows + 1, dtype=torch.long)
    torch.cumsum(counts, 0, out=crow[1:])
    col = torch.arange(nnz, dtype=torch.long) - torch.repeat_interleave(
        crow[:-1], counts
    )
    if batch:
        crow = crow.expand(batch + (rows + 1,)).contiguous()
        col = col.expand(batch + (nnz,)).contiguous()
    crow_t = crow.to(device=device, dtype=torch.long)
    col_t = col.to(device=device, dtype=torch.long)
    values_t = torch.randn(batch + (nnz,), dtype=dtype, generator=gen).to(device)
    return crow_t, col_t, values_t


def _case_fn(shape, dtype):
    del dtype
    tensor_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": tensor_shape},
        params={"nnz": nnz},
        builder_args=(tensor_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    tensor_shape, nnz = plan.builder_args
    crow, col, values = _make_csr_inputs(tensor_shape, nnz, dtype, device)
    # The trailing dict is unpacked into kwargs by the benchmark runner, so
    # torch_op and gems_op both receive (crow, col, values, size, dtype=...,
    # device=...). The dtype keyword is required (the aten factory forces
    # float32 storage otherwise) and the device keyword is required on CUDA.
    return crow, col, values, list(tensor_shape), {"dtype": dtype, "device": device}


class SparseCsrTensorBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark feeding the raw CSR components to the
    ``sparse_csr_tensor`` factory call."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_SHAPES)


@pytest.mark.sparse_csr_tensor
def test_sparse_csr_tensor():
    bench = SparseCsrTensorBenchmark(
        op_name="sparse_csr_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_csr_tensor,
        gems_op=getattr(flag_gems, "sparse_csr_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
