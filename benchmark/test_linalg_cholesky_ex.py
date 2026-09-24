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

# The 17 descriptors this operator is benchmarked on, restored in full: square
# 2-D sizes plus batched square sizes that reach the large-square factor paths.
_CHOLESKY_SHAPES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (4, 16, 16),
    (8, 32, 32),
    (8, 64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (4, 256, 256),
    (2, 512, 512),
    (4, 1024, 1024),
]

# A capability field, not a callable. FP64 and therefore complex128 are only
# benchmarked where the device reports support. The native kernel selection
# rejects fp16, bf16, fp8 and the integer dtypes, so they are not benchmarked.
_BENCH_DTYPES = [torch.float32] + list(consts.COMPLEX_DTYPES)
if flag_gems.runtime.device.support_fp64:
    _BENCH_DTYPES += [torch.float64, torch.complex128]


def _validate_shape(shape):
    """Reject descriptors this operator cannot factor, at planning time only."""
    if len(shape) < 2:
        raise ValueError(
            "linalg_cholesky_ex needs a square matrix, got rank " + str(len(shape))
        )
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("shape entries must be integers, got " + repr(dim))
        if dim < 0:
            raise ValueError("shape entries must be non-negative, got " + repr(dim))
    if shape[-1] != shape[-2]:
        raise ValueError(
            "linalg_cholesky_ex needs a square trailing pair, got " + str(shape)
        )


def _case_fn(shape, dtype):
    # Planning stage only: metadata is emitted without allocating any tensor,
    # and both upper settings are timed for every shape.
    del dtype
    _validate_shape(shape)
    for upper in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"input": tuple(shape)},
            params={"upper": upper},
            builder_args=(tuple(shape), upper),
        )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    n = shape[-1]
    # The operand is built here so the same construction covers every
    # benchmarked dtype, float64 and complex128 included.
    raw = torch.randn(*shape, dtype=dtype, device=device)
    gram = raw @ raw.conj().transpose(-2, -1)
    # The largest eigenvalue of an iid Gram matrix grows like 4 * n, so an 8 * n
    # shift makes the operand definite for every benchmarked shape and dtype
    # rather than letting the factorization fail silently.
    eye = torch.eye(n, dtype=dtype, device=device)
    matrix = gram + 8.0 * max(float(n), 1.0) * eye
    return matrix, {"upper": plan.params["upper"]}


class LinalgCholeskyExBenchmark(OperatorBenchmark):
    """Two-phase benchmark for a factorization that needs a square operand.

    The generic core and comprehensive shape sets contain rank < 2 and
    non-square entries that this operator rejects, so the square descriptors
    above are the defaults while a caller-supplied shape file still wins. Every
    shape is timed for both upper settings, which are the two kernel paths of
    this operator.
    """

    def set_shapes(self, shape_file_path=None):
        if shape_file_path is None:
            # No shape file: the built-in square descriptors are the defaults.
            self.shapes = [tuple(shape) for shape in _CHOLESKY_SHAPES]
            return
        super().set_shapes(shape_file_path, default_shapes=_CHOLESKY_SHAPES)

    def set_more_shapes(self):
        # The generic extra shapes are rank-1 or non-square, so the square
        # descriptors above stay the complete valid set.
        return []


@pytest.mark.linalg_cholesky_ex
def test_linalg_cholesky_ex():
    bench = LinalgCholeskyExBenchmark(
        op_name="linalg_cholesky_ex",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.linalg_cholesky_ex,
        gems_op=getattr(flag_gems, "linalg_cholesky_ex", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
