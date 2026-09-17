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
from . import conftest as cfg

# shifted_chebyshev_polynomial_t has no eager Half/BFloat16 CUDA kernel, so the
# supported set is the float32/float64 subset of utils.FLOAT_DTYPES.
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = [
        dtype
        for dtype in (*utils.FLOAT_DTYPES, torch.float64)
        if dtype in (torch.float32, torch.float64)
    ]


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_shifted_chebyshev_polynomial_t(shape, dtype, caplog):
    # x in [0, 1] for shifted Chebyshev polynomial
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    n = torch.randint(0, 10, shape, dtype=torch.long, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_n = n.to(ref_x.device).to(ref_x.dtype)

    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(ref_x, ref_n)
    logger_name = "flag_gems.ops.special_shifted_chebyshev_polynomial_t"
    with caplog.at_level("DEBUG", logger=logger_name):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(x, n)
    assert "GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_T" in caplog.text

    # Use larger tolerance for float32 due to trigonometric function precision
    utils.gems_assert_close(res_out, ref_out, dtype, atol=5e-3)


@pytest.mark.special_shifted_chebyshev_polynomial_t
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_special_shifted_chebyshev_polynomial_t_scalar_n(shape, dtype, caplog):
    # Scalar n reaches the same wrapper: aten decomposes the .n_scalar overload
    # onto the registered .default overload, so no separate registration exists.
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    n = 3  # scalar

    ref_x = utils.to_reference(x, True)

    ref_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(ref_x, n)
    logger_name = "flag_gems.ops.special_shifted_chebyshev_polynomial_t"
    with caplog.at_level("DEBUG", logger=logger_name):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_shifted_chebyshev_polynomial_t(x, n)
    assert "GEMS SPECIAL_SHIFTED_CHEBYSHEV_POLYNOMIAL_T" in caplog.text

    utils.gems_assert_close(res_out, ref_out, dtype)
