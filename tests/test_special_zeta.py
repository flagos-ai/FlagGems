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

from contextlib import contextmanager

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_ZETA_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _ZETA_DTYPES.append(torch.float64)


def _assert_close(actual, expected, dtype, *, atol=None):
    actual = utils.to_cpu(actual, expected)
    if dtype == torch.float64:
        rtol = 1e-12
        default_atol = 1e-14
    else:
        rtol = 2e-5
        default_atol = 2e-6
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=default_atol if atol is None else atol,
        equal_nan=True,
    )


def _reference(x, q):
    x_ref = utils.to_reference(x)
    q_ref = utils.to_reference(q)
    return torch.special.zeta(x_ref, q_ref)


@contextmanager
def _registered_special_zeta():
    library = torch.library.Library("aten", "IMPL")
    previous_registrar = flag_gems.current_work_registrar
    try:
        flag_gems.only_enable(
            lib=library,
            include=["special_zeta"],
            registrar=flag_gems.GeneralOpRegistrar,
        )
        yield
    finally:
        if hasattr(library, "_destroy"):
            library._destroy()
        flag_gems.current_work_registrar = previous_registrar


@pytest.mark.special_zeta
@pytest.mark.parametrize("dtype", _ZETA_DTYPES)
def test_special_zeta_broadcast(dtype):
    x = torch.rand((3, 1, 5), dtype=dtype, device=flag_gems.device) * 4.0 + 1.05
    q = torch.rand((1, 4, 1), dtype=dtype, device=flag_gems.device) * 6.0 + 0.1
    expected = _reference(x, q)

    actual = flag_gems.special_zeta(x, q)

    assert actual.shape == (3, 4, 5)
    assert actual.dtype == dtype
    _assert_close(actual, expected, dtype)


@pytest.mark.special_zeta_out
@pytest.mark.parametrize("dtype", _ZETA_DTYPES)
def test_special_zeta_out(dtype):
    x = torch.rand((3, 1, 5), dtype=dtype, device=flag_gems.device) * 4.0 + 1.05
    q = torch.rand((1, 4, 1), dtype=dtype, device=flag_gems.device) * 6.0 + 0.1
    expected = _reference(x, q)
    out = torch.empty((3, 4, 5), dtype=dtype, device=flag_gems.device)

    returned = flag_gems.special_zeta_out(x, q, out)

    assert returned is out
    _assert_close(out, expected, dtype)


@pytest.mark.special_zeta_tensor_scalar
@pytest.mark.parametrize("dtype", _ZETA_DTYPES)
def test_special_zeta_tensor_scalar(dtype):
    x = torch.linspace(1.05, 8.0, 37, dtype=dtype, device=flag_gems.device)
    q = 2.25
    expected = torch.special.zeta(utils.to_reference(x), q)

    actual = flag_gems.special_zeta_tensor_scalar(x, q)

    _assert_close(actual, expected, dtype)


@pytest.mark.special_zeta_tensor_scalar_out
@pytest.mark.parametrize("dtype", _ZETA_DTYPES)
def test_special_zeta_tensor_scalar_out(dtype):
    x = torch.linspace(1.05, 8.0, 37, dtype=dtype, device=flag_gems.device)
    q = 2.25
    expected = torch.special.zeta(utils.to_reference(x), q)
    out = torch.empty_like(x)

    returned = flag_gems.special_zeta_tensor_scalar_out(x, q, out)

    assert returned is out
    _assert_close(out, expected, dtype)


@pytest.mark.special_zeta_scalar_tensor
@pytest.mark.parametrize("dtype", _ZETA_DTYPES)
def test_special_zeta_scalar_tensor(dtype):
    x = 3.5
    q = torch.linspace(0.1, 20.0, 37, dtype=dtype, device=flag_gems.device)
    expected = torch.special.zeta(x, utils.to_reference(q))

    actual = flag_gems.special_zeta_scalar_tensor(x, q)

    _assert_close(actual, expected, dtype)


@pytest.mark.special_zeta_scalar_tensor_out
@pytest.mark.parametrize("dtype", _ZETA_DTYPES)
def test_special_zeta_scalar_tensor_out(dtype):
    x = 3.5
    q = torch.linspace(0.1, 20.0, 37, dtype=dtype, device=flag_gems.device)
    expected = torch.special.zeta(x, utils.to_reference(q))
    out = torch.empty_like(q)

    returned = flag_gems.special_zeta_scalar_tensor_out(x, q, out)

    assert returned is out
    _assert_close(out, expected, dtype)


@pytest.mark.special_zeta
@pytest.mark.parametrize("dtype", _ZETA_DTYPES)
def test_special_zeta_special_values(dtype):
    x = torch.tensor(
        [1.0, 0.5, 2.5, 2.0, 3.0, 2.0, float("inf"), float("inf"), float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    q = torch.tensor(
        [float("nan"), 2.0, 0.0, -1.0, -0.5, -0.5, 0.5, 1.0, 2.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    expected = _reference(x, q)

    actual = flag_gems.special_zeta(x, q)

    _assert_close(actual, expected, dtype, atol=1e-4)


@pytest.mark.special_zeta
@pytest.mark.parametrize("dtype", _ZETA_DTYPES)
def test_special_zeta_numerical_regimes(dtype):
    near_one = 1e-12 if dtype == torch.float64 else 1e-5
    x = torch.tensor(
        [1.0 + near_one, 1.001, 2.0, 10.0, 50.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    q = torch.tensor(
        [0.25, 1000.0, 100000.0, 10.0, 0.75],
        dtype=dtype,
        device=flag_gems.device,
    )
    expected = _reference(x, q)

    actual = flag_gems.special_zeta(x, q)

    _assert_close(actual, expected, dtype, atol=2e-5)


@pytest.mark.special_zeta
@pytest.mark.parametrize("dtype", _ZETA_DTYPES)
def test_special_zeta_negative_noninteger_q(dtype):
    x = torch.tensor([2.0, 3.0, 4.0, 2.5], dtype=dtype, device=flag_gems.device)
    q = torch.tensor([-0.5, -1.5, -6.5, -0.5], dtype=dtype, device=flag_gems.device)
    expected = _reference(x, q)

    actual = flag_gems.special_zeta(x, q)

    _assert_close(actual, expected, dtype, atol=2e-4)


@pytest.mark.special_zeta
def test_special_zeta_integer_promotion():
    x = torch.tensor([2, 3, 4], dtype=torch.int32, device=flag_gems.device)
    q = torch.tensor([1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    expected = _reference(x, q)

    actual = flag_gems.special_zeta(x, q)

    assert actual.dtype == torch.get_default_dtype()
    _assert_close(actual, expected, actual.dtype)


@pytest.mark.special_zeta
def test_special_zeta_half_bfloat16_promotion():
    if not flag_gems.runtime.device.support_bf16:
        pytest.skip("bfloat16 is not supported on this backend")
    x = torch.tensor([2.0, 3.0], dtype=torch.float16, device=flag_gems.device)
    q = torch.tensor([1.5, 2.5], dtype=torch.bfloat16, device=flag_gems.device)
    expected = _reference(x, q)

    actual = flag_gems.special_zeta(x, q)

    assert actual.dtype == torch.float32
    _assert_close(actual, expected, torch.float32)


@pytest.mark.special_zeta
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_special_zeta_rejects_native_low_precision(dtype):
    if dtype == torch.bfloat16 and not flag_gems.runtime.device.support_bf16:
        pytest.skip("bfloat16 is not supported on this backend")
    x = torch.tensor([2.0, 3.0], dtype=dtype, device=flag_gems.device)
    q = torch.tensor([1.5, 2.5], dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="float32 or float64"):
        flag_gems.special_zeta(x, q)


@pytest.mark.special_zeta_out
@pytest.mark.parametrize("alias", ["x", "q"])
def test_special_zeta_out_exact_alias(alias):
    x = torch.tensor([2.0, 3.0, 4.0], device=flag_gems.device)
    q = torch.tensor([1.5, 2.5, 3.5], device=flag_gems.device)
    expected = _reference(x, q)
    out = x if alias == "x" else q
    storage_ptr = out.untyped_storage().data_ptr()

    returned = flag_gems.special_zeta_out(x, q, out)

    assert returned is out
    assert out.untyped_storage().data_ptr() == storage_ptr
    _assert_close(out, expected, torch.float32)


@pytest.mark.special_zeta_out
def test_special_zeta_out_partial_overlap_rejected():
    storage = torch.linspace(1.5, 5.5, 8, device=flag_gems.device)
    x = storage[:4]
    q = torch.full_like(x, 2.0)
    out = storage[1:5]

    with pytest.raises(RuntimeError, match="single memory location"):
        flag_gems.special_zeta_out(x, q, out)


@pytest.mark.special_zeta_out
def test_special_zeta_out_casts_after_float_compute():
    x = torch.tensor([2.0, 3.0], dtype=torch.float32, device=flag_gems.device)
    q = torch.tensor([1.5, 2.5], dtype=torch.float32, device=flag_gems.device)
    expected = _reference(x, q).to(torch.float16)
    out = torch.empty(2, dtype=torch.float16, device=flag_gems.device)

    flag_gems.special_zeta_out(x, q, out)

    torch.testing.assert_close(utils.to_cpu(out, expected), expected, rtol=0, atol=0)


@pytest.mark.special_zeta
def test_special_zeta_q_gradient():
    x = torch.tensor([2.0, 3.0, 4.0], device=flag_gems.device)
    q = torch.tensor([1.5, 2.5, 3.5], device=flag_gems.device, requires_grad=True)
    x_ref = utils.to_reference(x)
    q_ref = utils.to_reference(q.detach())
    expected = -x_ref * torch.special.zeta(x_ref + 1.0, q_ref)

    with _registered_special_zeta():
        output = torch.special.zeta(x, q)
        (actual,) = torch.autograd.grad(output.sum(), q)

    _assert_close(actual, expected, torch.float32)


@pytest.mark.special_zeta
def test_special_zeta_x_gradient_is_not_implemented():
    x = torch.tensor([2.0, 3.0, 4.0], device=flag_gems.device, requires_grad=True)
    q = torch.tensor([1.5, 2.5, 3.5], device=flag_gems.device)

    with _registered_special_zeta():
        output = torch.special.zeta(x, q)
        with pytest.raises(RuntimeError, match="derivative.*zeta|zeta.*derivative"):
            output.sum().backward()
