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
from .generated_operator_utils import OperatorBenchmark

# float16, float32, bfloat16 and float64 all reach the SparseCsrCPU kernel; the
# shared float list is used as-is with float64 appended.
_BENCH_DTYPES = [*consts.FLOAT_DTYPES, torch.float64]

_CPU = torch.device("cpu")

# self is (M, K) and other is (K, N).
_SPARSE_MM_REDUCE_MNK = [
    (1024, 1024, 1024),
    (2048, 1024, 1024),
    (4096, 2048, 512),
    (16, 128, 64),
]

# Zero-extent descriptors: every one of them is a valid native call, so the same
# case function and builder are reused through a subclass that adds them.
_ZERO_EXTENT_MNK = [
    (0, 4, 3),
    (4, 0, 3),
    (4, 3, 0),
]

# The cost of this operator follows nnz * N, so the column step is the density
# axis and the four reduce modes are the semantic axis.
_VARIANTS = (
    ("sum", 16),
    ("sum", 32),
    ("mean", 16),
    ("amax", 16),
    ("amin", 16),
)


def _is_mnk(shape):
    """True for a descriptor this fixed-rank call can consume: exactly three
    non-negative integer extents (M, K, N)."""
    return (
        isinstance(shape, (list, tuple))
        and len(shape) == 3
        and all(
            isinstance(extent, int) and not isinstance(extent, bool) and extent >= 0
            for extent in shape
        )
    )


def _validate_mnk(triples):
    """Reject malformed or negative (M, K, N) descriptors with a clear error."""
    for triple in triples:
        if not _is_mnk(triple):
            raise ValueError(
                f"Invalid (M, K, N) descriptor {triple!r}: expected three "
                "non-negative integer extents."
            )


def _row_counts(m, k, step):
    """Stored-column count per row of the pattern built by ``_csr_operand``. Pure
    arithmetic so case listing can report the exact nnz without allocating."""
    if m == 0 or k == 0:
        return [0] * m
    return [(k - (row % step) + step - 1) // step for row in range(m)]


def _dense(shape, dtype):
    return torch.empty(shape, dtype=dtype, device=_CPU).uniform_(-1, 1)


def _csr_operand(m, k, step, dtype):
    """CSR operand storing every ``step``-th column of each row with a per-row
    offset, so nnz follows 1/step and the column indices differ per row."""
    if m == 0 or k == 0:
        return torch.sparse_csr_tensor(
            torch.zeros(m + 1, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=dtype),
            size=(m, k),
        )
    counts = torch.tensor(_row_counts(m, k, step), dtype=torch.int64)
    crow = torch.zeros(m + 1, dtype=torch.int64)
    crow[1:] = counts.cumsum(0)
    nnz = int(crow[-1])
    row_id = torch.repeat_interleave(torch.arange(m, dtype=torch.int64), counts)
    row_start = torch.repeat_interleave(crow[:-1], counts)
    starts = torch.arange(m, dtype=torch.int64) % step
    col = starts[row_id] + step * (torch.arange(nnz, dtype=torch.int64) - row_start)
    return torch.sparse_csr_tensor(crow, col, _dense((nnz,), dtype), size=(m, k))


def _case_fn(shape, dtype):
    # Metadata only: listing allocates no tensor and runs no operator.
    del dtype
    m, k, n = shape
    for reduce, step in _VARIANTS:
        yield base.BenchmarkCasePlan(
            shape={"self": (m, k), "other": (k, n)},
            params={
                "reduce": reduce,
                "nnz": sum(_row_counts(m, k, step)),
                "col_step": step,
                "layout": str(torch.sparse_csr),
            },
            builder_args=(m, k, n, reduce, step),
        )


def _build_inputs_fn(plan, dtype, device):
    m, k, n, reduce, step = plan.builder_args
    # The only registered kernel is SparseCsrCPU, so the benched call and the
    # reference call both receive CPU operands; the accelerator device is not part
    # of this operator's contract.
    del device
    self_t = _csr_operand(m, k, step, dtype)
    other = _dense((k, n), dtype)
    return self_t, other, {"reduce": reduce}


class SparseMmReduceImplBenchmark(OperatorBenchmark):
    """Two-phase cases for the CSR sparse-dense reduce (SparseCsrCPU only)."""

    DEFAULT_SHAPES = _SPARSE_MM_REDUCE_MNK
    DEFAULT_SHAPE_DESC = "M, K, N"

    def set_shapes(self, shape_file_path=None):
        # Normal loader semantics run first, so a core_shapes.yaml / --shape_file
        # entry for this operator still supplies workloads. core_shapes.yaml has no
        # such entry, so the loader falls back to its generic `Benchmark` class
        # entry, whose 1-D and 2-D sizes cannot express this fixed-rank call (self
        # is a 2-D CSR (M, K), other a 2-D dense (K, N)). Only 3-extent entries are
        # consumable and every one of them is kept -- nothing is dropped for cost,
        # and the two 3-extent entries of that generic list run as real workloads.
        _validate_mnk(self.DEFAULT_SHAPES)
        super().set_shapes(shape_file_path)
        loaded = [tuple(shape) for shape in self.shapes if _is_mnk(shape)]
        self.shapes = list(dict.fromkeys([*loaded, *self.DEFAULT_SHAPES]))


class SparseMmReduceImplZeroExtentBenchmark(SparseMmReduceImplBenchmark):
    """The same case pipeline with the zero-extent descriptors added."""

    DEFAULT_SHAPES = _ZERO_EXTENT_MNK


@pytest.mark.sparse_mm_reduce_impl
def test__sparse_mm_reduce_impl():
    bench = SparseMmReduceImplBenchmark(
        op_name="_sparse_mm_reduce_impl",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_mm_reduce_impl,
        gems_op=getattr(flag_gems, "_sparse_mm_reduce_impl", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()


@pytest.mark.sparse_mm_reduce_impl
def test__sparse_mm_reduce_impl_zero_extent():
    bench = SparseMmReduceImplZeroExtentBenchmark(
        op_name="_sparse_mm_reduce_impl",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_mm_reduce_impl,
        gems_op=getattr(flag_gems, "_sparse_mm_reduce_impl", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
