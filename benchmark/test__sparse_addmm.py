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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# pytest.mark cannot build a marker for an underscore-prefixed name through
# attribute access, so register it explicitly.
setattr(
    pytest.mark,
    "_sparse_addmm",
    MarkDecorator(Mark("_sparse_addmm", (), {}, _ispytest=True), _ispytest=True),
)

# The shared FLOAT_DTYPES list cannot be used as-is: 'addmm_sparse_cuda' has no
# Half and no BFloat16 kernel on this backend. float32 and float64 do have one,
# and float64 additionally needs the static capability flag, so case listing and
# case execution share this one dtype list.
_BENCH_DTYPES = [dtype for dtype in consts.FLOAT_DTYPES if dtype is torch.float32]
if flag_gems.runtime.device.support_fp64:
    _BENCH_DTYPES.append(torch.float64)

# _sparse_addmm is 2-D only (mat1.sparse_dim() == 2, mat2.dim() == 2) and
# core_shapes.yaml has no entry for this operator, so the generic default shape
# list is replaced by (M, N, K) triples of performance-relevant sizes.
_SPARSE_ADDMM_MNK = [
    (1024, 1024, 1024),
    (2048, 2048, 1024),
    (4096, 1024, 1024),
    (20, 320, 15),
    (16, 128, 64),
]

# Zero-extent descriptors: M == 0 and N == 0 produce an empty dense result, K == 0
# removes the product so the result is beta * self. Every one of them is a valid
# native call, so the same case function and builder are reused through a
# subclass that only changes this default.
_ZERO_EXTENT_MNK = [
    (0, 4, 3),
    (4, 0, 3),
    (4, 3, 0),
]

# Sparse-operand variants: a fully stored COO, genuinely sparse COO operands whose
# cost follows nnz instead of M*K, and CSR storage.
_LAYOUT_CASES = (
    ("coo", 1.0),
    ("coo", 0.1),
    ("coo", 0.01),
    ("csr", 0.01),
)


def _validate_mnk(triples):
    """Reject malformed or negative (M, N, K) descriptors with a clear error."""
    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            raise ValueError(
                f"Invalid (M, N, K) descriptor {triple!r}: expected three extents."
            )
        for extent in triple:
            if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
                raise ValueError(
                    f"Invalid (M, N, K) descriptor {triple!r}: extents must be "
                    "non-negative integers."
                )


def _dense(shape, dtype, device):
    """Dense operand built on the benchmark device for every benched dtype."""
    return torch.empty(shape, dtype=dtype, device=device).uniform_(-1, 1)


def _sparse_mat1(layout, density, shape, dtype, device):
    m, k = shape
    if density == 1.0:
        coo = _dense(shape, dtype, device).to_sparse()
    elif m * k == 0:
        # Zero capacity: the only consistent operand stores no coordinate at all,
        # so a placeholder stored value is never fabricated here.
        indices = torch.empty((2, 0), dtype=torch.int64, device=device)
        coo = torch.sparse_coo_tensor(
            indices,
            torch.empty(0, dtype=dtype, device=device),
            shape,
            dtype=dtype,
            device=device,
        )
    else:
        nnz = int(m * k * density)
        indices = torch.stack(
            [
                torch.randint(0, m, (nnz,), device=device),
                torch.randint(0, k, (nnz,), device=device),
            ]
        )
        coo = torch.sparse_coo_tensor(
            indices, _dense((nnz,), dtype, device), shape, dtype=dtype, device=device
        )
    coo = coo.coalesce()
    return coo.to_sparse_csr() if layout == "csr" else coo


def _case_fn(shape, dtype):
    # Metadata only: listing allocates no tensor and runs no operator.
    del dtype
    m, n, k = shape
    for layout, density in _LAYOUT_CASES:
        yield base.BenchmarkCasePlan(
            shape={"self": (m, n), "mat1": (m, k), "mat2": (k, n)},
            params={
                "beta": 1,
                "alpha": 1,
                "layout": str(
                    torch.sparse_coo if layout == "coo" else torch.sparse_csr
                ),
                "density": density,
            },
            builder_args=(m, n, k, layout, density),
        )


def _build_inputs_fn(plan, dtype, device):
    m, n, k, layout, density = plan.builder_args
    self_t = _dense((m, n), dtype, device)
    mat1 = _sparse_mat1(layout, density, (m, k), dtype, device)
    mat2 = _dense((k, n), dtype, device)
    return (
        self_t,
        mat1,
        mat2,
        {"beta": plan.params["beta"], "alpha": plan.params["alpha"]},
    )


class SparseAddmmBenchmark(OperatorBenchmark):
    """Two-phase cases restricted to valid 2-D sparse-addmm operands."""

    default_mnk = _SPARSE_ADDMM_MNK

    def set_shapes(self, shape_file_path=None, *, default_shapes=None):
        # A caller-supplied yaml entry for this operator still wins; otherwise the
        # triples above replace the generic default shape list. Descriptors taken
        # from either source are validated before use.
        triples = list(self.default_mnk if default_shapes is None else default_shapes)
        _validate_mnk(triples)
        super().set_shapes(shape_file_path, default_shapes=triples)
        _validate_mnk(self.shapes)


class SparseAddmmZeroExtentBenchmark(SparseAddmmBenchmark):
    """The same case pipeline driven by the zero-extent descriptors."""

    default_mnk = _ZERO_EXTENT_MNK


@pytest.mark._sparse_addmm
def test__sparse_addmm():
    bench = SparseAddmmBenchmark(
        op_name="_sparse_addmm",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_addmm,
        gems_op=getattr(flag_gems, "_sparse_addmm", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()


@pytest.mark._sparse_addmm
def test__sparse_addmm_zero_extent():
    bench = SparseAddmmZeroExtentBenchmark(
        op_name="_sparse_addmm",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_addmm,
        gems_op=getattr(flag_gems, "_sparse_addmm", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
