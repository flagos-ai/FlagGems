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

"""Independent residual and full-write checks for every benchmark configuration."""

import pytest
import torch

from .test_triangular_solve import (
    DTYPES,
    FLAGS,
    _assert_residual,
    _assert_result,
    _functional,
    _inputs,
    _out,
    _reference,
)

# Keep the complete benchmark matrix, including slow and degenerate rows.
SYSTEMS = [
    ((), (), 8, 1),
    ((), (), 8, 8),
    ((), (), 32, 1),
    ((), (), 32, 64),
    ((), (), 128, 1),
    ((), (), 128, 8),
    ((), (), 512, 8),
    ((), (), 512, 64),
    ((), (), 1024, 1),
    ((), (), 1024, 64),
    ((4,), (4,), 128, 8),
    ((2, 1), (3,), 32, 64),
]


@pytest.mark.triangular_solve
@pytest.mark.triangular_solve_out
@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_out", [False, True], ids=["default", "out"])
def test_triangular_solve_benchmark_matrix(system, flags, dtype, use_out):
    B, A = _inputs(*system, dtype, flags[2])
    reference = _reference(B, A, *flags)
    before_A, before_B = A.clone(), B.clone()
    if use_out:
        outputs = [
            torch.full(
                (*ref.shape[:-2], ref.shape[-1], ref.shape[-2]),
                float("nan"),
                dtype=dtype,
                device=A.device,
            ).mT
            for ref in reference
        ]
        X, M = outputs
        pointers, strides = (X.data_ptr(), M.data_ptr()), (X.stride(), M.stride())
        result = _out(B, A, flags, X, M)
        assert result[0] is X and result[1] is M
        assert (X.data_ptr(), M.data_ptr()) == pointers
        assert (X.stride(), M.stride()) == strides
    else:
        result = _functional(B, A, flags)
    _assert_result(result, reference, B, A, functional=not use_out)
    _assert_residual(result[0], B, A, flags)
    torch.testing.assert_close(A, before_A, atol=0, rtol=0)
    torch.testing.assert_close(B, before_B, atol=0, rtol=0)
