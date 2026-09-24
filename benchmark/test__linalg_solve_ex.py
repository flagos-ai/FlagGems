# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# Register the underscore-prefixed pytest marker explicitly.
setattr(
    pytest.mark,
    "_linalg_solve_ex",
    MarkDecorator(Mark("_linalg_solve_ex", (), {}, _ispytest=True), _ispytest=True),
)

# Measured with valid rank>=2 operands: float32, complex64, float64 and
# complex128 are solved natively. Only the 64-bit types follow the backend
# capability flag; complex64 does not need fp64 support.
SUPPORTED_DTYPES = [torch.float32, torch.complex64]
if flag_gems.runtime.device.support_fp64:
    SUPPORTED_DTYPES += [torch.float64, torch.complex128]

# Performance descriptors for the operator geometry (batch..., n, m): solvers
# are dominated by the n x n factorization, so the grid sweeps the matrix size,
# the right-hand-side width and the batch depth. No element or memory cap is
# applied.
PERFORMANCE_DESCRIPTORS = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 512),
    (16, 256, 128),
    (8, 4, 128, 64),
]

# Zero-extent operands are accepted natively, so the empty-system boundaries
# stay in the default grid (measured: A (0,0)/B (0,0), A (64,64)/B (64,0) and
# A (2,0,0)/B (2,0,0) return without error for both left modes).
ZERO_DESCRIPTORS = [(0, 0), (64, 0), (2, 0, 0)]


def _validate_descriptor(shape):
    """Reject anything that cannot describe an A of shape (batch..., n, n).

    Descriptors arrive from the default list or from a user shape file; both
    paths go through the case builder, so an invalid entry fails loudly instead
    of listing successfully and breaking later while building inputs.
    """
    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"Benchmark descriptor must be a tuple or list, got {shape!r}")
    if len(shape) < 2:
        raise ValueError(f"aten::_linalg_solve_ex needs rank >= 2, got {shape!r}")
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ValueError(f"Descriptor extents must be integers, got {shape!r}")
        if extent < 0:
            raise ValueError(f"Descriptor extents must be nonnegative, got {shape!r}")
    return tuple(shape)


def _operand_shapes(shape, left):
    """A/B shapes for one descriptor with every batch axis preserved.

    A is (batch..., n, n) with n = shape[-2]; B is (batch..., n, m) for
    AX = B and (batch..., m, n) for XA = B, with m = shape[-1].
    """
    batch = tuple(shape[:-2])
    n, m = shape[-2], shape[-1]
    return batch + (n, n), (batch + (n, m) if left else batch + (m, n))


def _case_fn(shape, dtype):
    del dtype
    descriptor = _validate_descriptor(shape)
    for left in (True, False):
        a_shape, b_shape = _operand_shapes(descriptor, left)
        yield base.BenchmarkCasePlan(
            shape={"A": a_shape, "B": b_shape},
            params={"left": left},
            builder_args=(descriptor, left),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, left = plan.builder_args
    a_shape, b_shape = _operand_shapes(shape, left)
    # generate_tensor_input only covers consts.FLOAT_DTYPES and returns None for
    # float64/complex128, so the operands are built directly; torch.randn
    # supports all four natively solved types.
    a = torch.randn(a_shape, dtype=dtype, device=device)
    a.diagonal(dim1=-2, dim2=-1).add_(a_shape[-1])
    b = torch.randn(b_shape, dtype=dtype, device=device)
    return a, b, {"left": left}


DEFAULT_DESCRIPTORS = [
    _validate_descriptor(shape) for shape in PERFORMANCE_DESCRIPTORS + ZERO_DESCRIPTORS
]


class SolveExBenchmark(OperatorBenchmark):
    """Two-phase benchmark using the operator's own operand geometry.

    GenericBenchmark's default shapes contain 0-dim and 1-dim tensors, which
    aten::_linalg_solve_ex rejects (A must be at least 2-D), so the default
    descriptor list is replaced while an explicitly configured --shape-file
    still takes precedence for this operator name and class key.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_DESCRIPTORS)


@pytest.mark._linalg_solve_ex
def test__linalg_solve_ex():
    bench = SolveExBenchmark(
        op_name="_linalg_solve_ex",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._linalg_solve_ex,
        gems_op=getattr(flag_gems, "_linalg_solve_ex", None),
        dtypes=SUPPORTED_DTYPES,
    )
    bench.run()
