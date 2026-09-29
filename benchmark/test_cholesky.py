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

# Performance shapes: 8x8 up to 1024x1024 plus batched matrices. Rank >= 2 with
# equal last two dimensions is the only domain aten::cholesky accepts, so the
# GenericBenchmark default shapes, which contain 1-D rows, cannot be used.
CHOLESKY_SHAPES = [
    (8, 8),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (16, 128, 128),
    (64, 256, 256),
]

# The factorizer implements float32/float64/complex64/complex128 only (measured on
# this backend), so consts.FLOAT_DTYPES, which also holds fp16 and bf16, is not
# the right set. The fp64 pair is gated by the static backend capability flag.
CHOLESKY_DTYPES = [torch.float32] + consts.COMPLEX_DTYPES
if flag_gems.runtime.device.support_fp64:
    CHOLESKY_DTYPES += [torch.float64, torch.complex128]


def _validate_shape(shape):
    # A descriptor the operator cannot use is reported while the case list is
    # planned, so listing and execution stay on the same plans instead of the
    # invalid row being silently dropped. Pure metadata: no tensors are built.
    if len(shape) < 2:
        raise ValueError(f"cholesky needs matrices or batches of matrices, got {shape}")
    if shape[-2] != shape[-1]:
        raise ValueError(f"cholesky needs square matrices, got {shape}")
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"cholesky needs non-negative integer dims, got {shape}")


def _case_fn(shape, dtype):
    del dtype
    _validate_shape(shape)
    # One plan per triangular form; both kernel paths are timed.
    for upper in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"upper": upper},
            builder_args=(shape, upper),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, upper = plan.builder_args
    n = shape[-1]
    # A = M @ M.mH + n * I is Hermitian positive definite for any M, which the
    # factorization requires. utils.generate_tensor_input has no float64 or
    # complex128 support, so the matrix is built directly here.
    m = torch.randn(shape, dtype=dtype, device=device)
    inp = m @ m.mH + n * torch.eye(n, dtype=dtype, device=device)
    return inp, {"upper": upper}


class CholeskyBenchmark(OperatorBenchmark):
    # The generic default shape set contains 1-D rows that aten::cholesky rejects,
    # so set_shapes replaces them with CHOLESKY_SHAPES through the documented
    # default_shapes argument, which still gives a caller-provided shape file
    # precedence over these defaults.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=CHOLESKY_SHAPES)


@pytest.mark.cholesky
def test_cholesky():
    bench = CholeskyBenchmark(
        op_name="cholesky",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.cholesky,
        gems_op=getattr(flag_gems, "cholesky", None),
        dtypes=CHOLESKY_DTYPES,
    )
    bench.run()
