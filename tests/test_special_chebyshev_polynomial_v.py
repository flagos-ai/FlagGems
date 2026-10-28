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


@pytest.mark.special_chebyshev_polynomial_v
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_v_tensor_degree(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device).clamp(-0.99, 0.99)
    degree = torch.randint(0, 5, shape, dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)

    ref_out = torch.ops.aten.special_chebyshev_polynomial_v.default(
        ref_x, ref_degree
    )
    res_out = flag_gems.special_chebyshev_polynomial_v(x, degree)

    assert res_out.shape == x.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_v
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_v_degree_boundaries(dtype):
    x = torch.tensor(
        [-0.99, -0.75, -0.25, 0.25, 0.75],
        dtype=dtype,
        device=flag_gems.device,
    )
    degree = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)

    ref_out = torch.ops.aten.special_chebyshev_polynomial_v.default(
        ref_x, ref_degree
    )
    res_out = flag_gems.special_chebyshev_polynomial_v(x, degree)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_v
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_v_registered_dispatch(dtype, caplog, request):
    x = torch.tensor(
        [-0.75, -0.25, 0.25, 0.75], dtype=dtype, device=flag_gems.device
    )
    degree = torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)
    ref_out = torch.ops.aten.special_chebyshev_polynomial_v.default(
        ref_x, ref_degree
    )

    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_chebyshev_polynomial_v"):
        res_out = flag_gems.special_chebyshev_polynomial_v(x, degree)

    registry = getattr(request.config, "_override_registry", None)
    if registry is None or registry.get_override("special_chebyshev_polynomial_v") is None:
        expected = (
            f"{utils.gems_log_prefix(flag_gems.special_chebyshev_polynomial_v)} "
            "SPECIAL_CHEBYSHEV_POLYNOMIAL_V"
        )
        assert expected in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
