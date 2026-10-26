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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


def _assert_out_dispatch(caplog, request):
    registry = getattr(request.config, "_override_registry", None)
    if (
        registry is not None
        and registry.get_override("special_chebyshev_polynomial_t_out") is not None
    ):
        # A supplied candidate need not emit the original implementation's log.
        return
    expected = (
        f"{utils.gems_log_prefix(flag_gems.special_chebyshev_polynomial_t_out)} "
        "SPECIAL_CHEBYSHEV_POLYNOMIAL_T_OUT"
    )
    assert expected in caplog.text


@pytest.mark.special_chebyshev_polynomial_t_out
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_out_scalar_degree(shape, dtype):
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 2 - 1
    ref_x = utils.to_reference(x, True)
    degree = 3
    ref_out = torch.special.chebyshev_polynomial_t(ref_x, degree)
    out = torch.empty_like(x)
    original_pointer = out.data_ptr()

    res_out = flag_gems.special_chebyshev_polynomial_t_out(x, degree, out)

    assert res_out is out
    assert res_out.data_ptr() == original_pointer
    assert res_out.shape == x.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t_out
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_chebyshev_polynomial_t_out_tensor_degree(dtype, caplog, request):
    x = torch.tensor([-0.75, -0.25, 0.25, 0.75], dtype=dtype, device=flag_gems.device)
    degree = torch.tensor([0, 1, 2, 5], dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)
    ref_out = torch.empty_like(ref_x)
    torch.ops.aten.special_chebyshev_polynomial_t.out(ref_x, ref_degree, out=ref_out)
    out = torch.empty_like(x)
    original_pointer = out.data_ptr()

    with caplog.at_level(
        "DEBUG", logger="flag_gems.ops.special_chebyshev_polynomial_t"
    ):
        res_out = flag_gems.special_chebyshev_polynomial_t_out(x, degree, out=out)

    _assert_out_dispatch(caplog, request)
    assert res_out is out
    assert res_out.data_ptr() == original_pointer
    assert res_out.shape == x.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t_out
@pytest.mark.parametrize("out_shape", [(0,), (1,)])
def test_special_chebyshev_polynomial_t_out_resizes_buffer(out_shape):
    if flag_gems.vendor_name == "ascend" and not utils.TO_CPU and out_shape == (1,):
        pytest.skip("Ascend native reference cannot resize a nonempty out buffer")
    x = torch.tensor(
        [[-0.75, -0.25, 0.25], [0.5, 0.75, 1.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    degree = torch.tensor(3, dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_degree = utils.to_reference(degree, False)
    ref_out = torch.empty(out_shape, dtype=ref_x.dtype, device=ref_x.device)
    torch.ops.aten.special_chebyshev_polynomial_t.out(ref_x, ref_degree, out=ref_out)
    out = torch.empty(out_shape, dtype=x.dtype, device=flag_gems.device)

    res_out = flag_gems.special_chebyshev_polynomial_t_out(x, degree, out=out)

    assert res_out is out
    assert res_out.shape == x.shape
    assert res_out.dtype == x.dtype
    utils.gems_assert_close(res_out, ref_out, x.dtype, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t_out
def test_special_chebyshev_polynomial_t_out_accepts_float16_buffer():
    x = torch.tensor(
        [-0.75, -0.25, 0.25, 0.75],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    degree = torch.tensor([0, 1, 2, 5], dtype=torch.int64, device=flag_gems.device)
    ref_x = utils.to_reference(x, False)
    ref_degree = utils.to_reference(degree, False)
    ref_out = torch.empty(ref_x.shape, dtype=torch.float16, device=ref_x.device)
    torch.ops.aten.special_chebyshev_polynomial_t.out(ref_x, ref_degree, out=ref_out)
    out = torch.empty(x.shape, dtype=torch.float16, device=flag_gems.device)
    original_pointer = out.data_ptr()

    res_out = flag_gems.special_chebyshev_polynomial_t_out(x, degree, out=out)

    assert res_out is out
    assert res_out.data_ptr() == original_pointer
    assert res_out.dtype == torch.float16
    utils.gems_assert_close(res_out, ref_out, torch.float16, equal_nan=True)


@pytest.mark.special_chebyshev_polynomial_t_out
def test_special_chebyshev_polynomial_t_out_rejects_integer_buffer():
    x = torch.tensor(
        [-0.75, -0.25, 0.25, 0.75],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    degree = torch.tensor([0, 1, 2, 5], dtype=torch.int64, device=flag_gems.device)
    out = torch.empty(x.shape, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="cast|dtype|type"):
        flag_gems.special_chebyshev_polynomial_t_out(x, degree, out=out)


@pytest.mark.special_chebyshev_polynomial_t_out
def test_special_chebyshev_polynomial_t_out_rejects_unsupported_dtype():
    x = torch.randint(0, 4, (16,), dtype=torch.int32, device=flag_gems.device)
    out = torch.empty((16,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(ValueError, match="only supports float32/float64"):
        flag_gems.special_chebyshev_polynomial_t_out(x, 3, out)
