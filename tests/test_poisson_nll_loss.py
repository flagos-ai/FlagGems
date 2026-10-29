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

SHAPES = [(2, 3)] if cfg.QUICK_MODE else [(2, 3), (128, 256), (512, 512)]


def _poisson_ones(shape, dtype):
    if (flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64) or (
        flag_gems.vendor_name == "enflame" and dtype in (torch.int64, torch.float64)
    ):
        value = torch.ones(shape, dtype=dtype, device="cpu")
        return (
            value.reshape(-1)
            .view(torch.uint8)
            .to(flag_gems.device)
            .view(dtype)
            .reshape(shape)
        )
    return torch.ones(shape, dtype=dtype, device=flag_gems.device)


def _poisson_reference_input(value):
    if utils.TO_CPU and (
        (flag_gems.vendor_name == "kunlunxin" and value.dtype == torch.float64)
        or (
            flag_gems.vendor_name == "enflame"
            and value.dtype in (torch.int64, torch.float64)
        )
    ):
        return (
            value.reshape(-1)
            .view(torch.uint8)
            .cpu()
            .view(value.dtype)
            .reshape(value.shape)
        )
    return utils.to_reference(value)


def _complex_reference(inp, target, log_input, full, eps, reduction):
    try:
        return torch.ops.aten.poisson_nll_loss(
            inp, target, log_input, full, eps, reduction
        )
    except (RuntimeError, ValueError) as error:
        if (
            flag_gems.vendor_name == "kunlunxin"
            and inp.device.type != "cpu"
            and (inp.is_complex() or target.is_complex())
            and str(error).startswith("[NOT IMPLEMENTED]: error code= 4(")
        ):
            pytest.skip(
                "Kunlunxin native Poisson NLL does not support this complex configuration; CPU reference remains covered"
            )
        raise


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("log_input", [True, False])
@pytest.mark.parametrize("full", [False, True])
def test_accuracy_poisson_nll_loss(shape, dtype, reduction, log_input, full):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if not log_input:
        input = input.abs() + 0.1
    target = torch.randint(0, 5, shape, device=flag_gems.device).to(dtype)
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input, upcast=True),
        utils.to_reference(target, upcast=True),
        log_input,
        full,
        1e-8,
        reduction,
    )
    result = flag_gems.poisson_nll_loss(input, target, log_input, full, 1e-8, reduction)
    utils.gems_assert_close(result, ref, dtype, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("reduction", [0, 1, 2, 3])
def test_accuracy_poisson_nll_loss_broadcast_noncontiguous(reduction):
    input = torch.randn((7, 5), device=flag_gems.device).T
    target = torch.randint(0, 5, (1, 7), device=flag_gems.device).float()
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input),
        utils.to_reference(target),
        True,
        True,
        1e-8,
        reduction,
    )
    result = flag_gems.poisson_nll_loss(input, target, True, True, 1e-8, reduction)
    utils.gems_assert_close(result, ref, torch.float32, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_accuracy_poisson_nll_loss_empty(reduction):
    if (
        flag_gems.vendor_name == "enflame"
        and not utils.TO_CPU
        and reduction == 1
    ):
        pytest.skip(
            "Enflame native Poisson NLL aborts for empty mean reduction; "
            "CPU reference remains covered"
        )
    input = torch.empty((0, 3), device=flag_gems.device)
    target = torch.empty_like(input)
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input),
        utils.to_reference(target),
        True,
        False,
        1e-8,
        reduction,
    )
    result = flag_gems.poisson_nll_loss(input, target, True, False, 1e-8, reduction)
    if flag_gems.vendor_name == "enflame" and not utils.TO_CPU and reduction == 2:
        assert result.device == ref.device == input.device
        assert result.dtype == ref.dtype == torch.float32
        assert result.shape == ref.shape == torch.Size([])
        result, ref = result.cpu(), ref.cpu()
    utils.gems_assert_close(result, ref, torch.float32, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_poisson_nll_loss_boundary(dtype):
    input = torch.tensor(
        [0.0, -0.0, 1e-5, float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    target = torch.tensor([1.0, 0.0, 2.0, 0.0, 1.0], dtype=dtype, device=input.device)
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input, upcast=True),
        utils.to_reference(target, upcast=True),
        False,
        True,
        1e-8,
        0,
    )
    result = flag_gems.poisson_nll_loss(input, target, False, True, 1e-8, 0)
    utils.gems_assert_close(result, ref, dtype, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize(
    "input_dtype,target_dtype,expected_dtype",
    [
        (torch.int64, torch.float16, torch.float32),
        (torch.float16, torch.int64, torch.float16),
        (torch.bool, torch.float32, torch.float32),
        (torch.float32, torch.bool, torch.float32),
        (torch.float32, torch.float64, torch.float64),
    ],
)
def test_poisson_nll_loss_mixed_dtype_promotion(
    input_dtype, target_dtype, expected_dtype
):
    input = _poisson_ones((2, 3), input_dtype)
    target = _poisson_ones((1, 3), target_dtype)
    if (
        flag_gems.vendor_name == "kunlunxin"
        and not utils.TO_CPU
        and target_dtype == torch.float64
    ):
        pytest.skip(
            "Kunlunxin native Poisson NLL rejects the mixed float32/float64 configuration; CPU reference remains covered"
        )
    if (
        flag_gems.vendor_name == "enflame"
        and not utils.TO_CPU
        and (input_dtype, target_dtype)
        in (
            (torch.bool, torch.float32),
            (torch.int64, torch.float16),
            (torch.float16, torch.int64),
            (torch.float32, torch.float64),
        )
    ):
        pytest.skip(
            "Enflame native Poisson NLL does not support this mixed "
            "dtype configuration; CPU reference remains covered"
        )
    ref = torch.ops.aten.poisson_nll_loss(
        _poisson_reference_input(input),
        _poisson_reference_input(target),
        True,
        False,
        1e-8,
        0,
    )
    result = flag_gems.poisson_nll_loss(input, target, True, False, 1e-8, 0)
    assert result.device == input.device
    assert result.shape == ref.shape
    assert result.dtype == expected_dtype
    assert ref.dtype == expected_dtype
    if (
        utils.TO_CPU
        and flag_gems.vendor_name in ("kunlunxin", "enflame")
        and expected_dtype == torch.float64
    ):
        observed = (
            result.reshape(-1)
            .view(torch.uint8)
            .cpu()
            .view(result.dtype)
            .reshape(result.shape)
        )
        utils.gems_assert_close(observed, ref.cpu(), expected_dtype)
    else:
        utils.gems_assert_close(result, ref, expected_dtype)


@pytest.mark.poisson_nll_loss
def test_poisson_nll_loss_mixed_dtype_exceptions():
    if flag_gems.vendor_name == "enflame" and not utils.TO_CPU:
        pytest.skip(
            "Enflame native Poisson NLL does not support bool exp; "
            "CPU reference remains covered"
        )
    bool_input = torch.ones((2, 3), dtype=torch.bool, device=flag_gems.device)
    float_input = torch.ones((2, 3), device=flag_gems.device)
    for inp, target, full in (
        (bool_input, bool_input, False),
        (float_input, bool_input, True),
    ):
        # Native backends differ on bool acceptance. Check their actual contract
        # in both reference modes rather than failing before candidate invocation.
        try:
            reference = torch.ops.aten.poisson_nll_loss(
                utils.to_reference(inp), utils.to_reference(target), True, full, 1e-8, 0
            )
        except RuntimeError:
            with pytest.raises(RuntimeError):
                flag_gems.poisson_nll_loss(inp, target, True, full, 1e-8, 0)
        else:
            result = flag_gems.poisson_nll_loss(inp, target, True, full, 1e-8, 0)
            assert result.device == inp.device
            utils.gems_assert_close(result, reference, reference.dtype, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_poisson_nll_loss_bool_input_nonlog_rejected(full, reduction):
    if flag_gems.vendor_name == "ascend" and not utils.TO_CPU:
        pytest.skip("Ascend Poisson NLL reference accepts non-logarithmic bool input")
    inp = torch.ones((2, 3), dtype=torch.bool, device=flag_gems.device)
    target = torch.ones((1, 3), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.poisson_nll_loss(
            utils.to_reference(inp),
            utils.to_reference(target),
            False,
            full,
            1e-8,
            reduction,
        )
    with pytest.raises(RuntimeError):
        flag_gems.poisson_nll_loss(inp, target, False, full, 1e-8, reduction)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("log_input", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize(
    "case", ["broadcast_conjugate", "large", "empty", "real_input", "real_target"]
)
def test_poisson_nll_loss_complex(dtype, log_input, reduction, case):
    if dtype == torch.complex128 and not utils.fp64_is_supported:
        pytest.skip("FP64 is not supported")
    if (
        flag_gems.vendor_name == "enflame"
        and not utils.TO_CPU
        and not utils.complex_is_supported
    ):
        pytest.skip(
            "Enflame target does not support native complex compute; "
            "CPU reference remains covered"
        )
    if flag_gems.vendor_name == "mthreads" and not utils.TO_CPU and case != "empty":
        pytest.skip(
            "MUSA Poisson NLL reference does not support nonempty complex tensors"
        )
    if (
        flag_gems.vendor_name == "ascend"
        and not utils.TO_CPU
        and case == "empty"
        and reduction == 1
    ):
        pytest.skip("Ascend native empty complex mean does not support this dtype")
    shape = (65537,) if case == "large" else (0, 3) if case == "empty" else (3, 7)
    factory_device = (
        "cpu"
        if flag_gems.vendor_name in ("ascend", "enflame")
        else flag_gems.device
    )
    inp = torch.randn(shape, dtype=dtype, device=factory_device).to(flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=factory_device).to(flag_gems.device)
    if case == "broadcast_conjugate":
        inp = inp.T.conj()
        target = target[:1].T.conj()
    elif case == "real_input":
        inp = inp.real.abs() + 0.5
    elif case == "real_target":
        target = target.real.abs() + 0.5
    full = case == "real_target"
    reference = _complex_reference(
        utils.to_reference(inp),
        utils.to_reference(target),
        log_input,
        full,
        1e-8,
        reduction,
    )
    result = flag_gems.poisson_nll_loss(inp, target, log_input, full, 1e-8, reduction)
    assert result.device == inp.device
    assert result.dtype == reference.dtype
    assert result.shape == reference.shape
    if flag_gems.vendor_name in ("ascend", "enflame") or (
        flag_gems.vendor_name == "kunlunxin" and result.is_complex()
    ):
        result, reference = result.cpu(), reference.cpu()
    utils.gems_assert_close(
        result,
        reference,
        dtype,
        equal_nan=True,
        reduce_dim=max(inp.numel(), 1) if reduction in (1, 2) else 1,
    )


@pytest.mark.poisson_nll_loss
def test_poisson_nll_loss_complex_target_full_rejected():
    if (
        flag_gems.vendor_name == "enflame"
        and not utils.TO_CPU
        and not utils.complex_is_supported
    ):
        pytest.skip(
            "Enflame target does not support native complex compute; "
            "CPU reference remains covered"
        )
    inp = torch.ones(3, dtype=torch.complex64, device="cpu").to(flag_gems.device)
    with pytest.raises((RuntimeError, ValueError)) as error:
        torch.ops.aten.poisson_nll_loss(
            utils.to_reference(inp), utils.to_reference(inp), True, True, 1e-8, 0
        )
    if isinstance(error.value, ValueError):
        assert flag_gems.vendor_name == "kunlunxin" and not utils.TO_CPU
        assert str(error.value).startswith("[NOT IMPLEMENTED]: error code= 4(")
    with pytest.raises(RuntimeError):
        flag_gems.poisson_nll_loss(inp, inp, True, True, 1e-8, 0)


@pytest.mark.poisson_nll_loss
def test_poisson_nll_loss_invalid_broadcast():
    input = torch.ones((2, 3), device=flag_gems.device)
    target = torch.ones((4, 3), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.poisson_nll_loss(input, target, True, False, 1e-8, 0)
    with pytest.raises(RuntimeError):
        flag_gems.poisson_nll_loss(input, target, True, False, 1e-8, 0)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("eps", [0.0, -1e-3, float("inf"), float("nan")])
def test_poisson_nll_loss_special_eps(eps):
    input = torch.tensor([0.0, 0.5, 2.0], device=flag_gems.device)
    target = torch.tensor([1.0, 2.0, 3.0], device=flag_gems.device)
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input),
        utils.to_reference(target),
        False,
        False,
        eps,
        0,
    )
    result = flag_gems.poisson_nll_loss(input, target, False, False, eps, 0)
    utils.gems_assert_close(result, ref, torch.float32, equal_nan=True)


@pytest.mark.poisson_nll_loss
@pytest.mark.parametrize("size", [65536, 65537])
@pytest.mark.parametrize("reduction", [1, 2])
def test_poisson_nll_loss_reduction_boundary(size, reduction):
    input = torch.randn(size, device=flag_gems.device)
    target = torch.randint(0, 5, (size,), device=flag_gems.device).float()
    ref = torch.ops.aten.poisson_nll_loss(
        utils.to_reference(input, upcast=True),
        utils.to_reference(target, upcast=True),
        True,
        False,
        1e-8,
        reduction,
    )
    result = flag_gems.poisson_nll_loss(input, target, True, False, 1e-8, reduction)
    utils.gems_assert_close(result, ref, torch.float32, reduce_dim=size, equal_nan=True)
