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

from . import accuracy_utils as utils

MATRIX_POWER_EXPONENTS = [0, 1, 2, 3, 5, 8]

MATRIX_POWER_SHAPES = [
    ((), 2),
    ((), 8),
    ((), 32),
    ((4,), 16),
    ((2, 3), 8),
]

def _make_input(shape, dim, dtype):
    # Build a well-conditioned matrix close to the identity so that repeated
    # squaring keeps element magnitudes bounded. Unit-variance gaussians would
    # blow up geometrically with n and lose all precision in low-precision dtypes.
    eye = torch.eye(dim, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.1 + eye

@pytest.mark.matrix_power_out
@pytest.mark.parametrize("batch_shape, dim", MATRIX_POWER_SHAPES)
@pytest.mark.parametrize("n", MATRIX_POWER_EXPONENTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matrix_power_out(batch_shape, dim, n, dtype):
    torch.manual_seed(0)
    shape = (*batch_shape, dim, dim)
    res_inp = _make_input(shape, dim, dtype)
    ref_inp = utils.to_reference(res_inp)

    if n == 0:
        # An identity is exact on the CPU, including for batched matrices.
        ref_out = torch.matrix_power(ref_inp.cpu(), n).to(ref_inp.device)
    else:
        ref_out = torch.matrix_power(ref_inp, n)

    res_out = torch.empty_like(res_inp)
    out_pointer = res_out.data_ptr()
    result = flag_gems.matrix_power_out(res_inp, n, out=res_out)

    assert result is res_out
    assert result.device == res_inp.device
    assert res_out.data_ptr() == out_pointer
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim * max(n, 1))
