import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


if flag_gems.vendor_name == "nvidia":
    DEVICE_AVAILABLE = torch.cuda.is_available()
elif flag_gems.vendor_name == "ascend":
    DEVICE_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
else:
    DEVICE_AVAILABLE = False

SUPPORTED_DTYPES = [torch.float32, torch.complex64]
COMPLEX_DTYPES = [torch.complex64]
if flag_gems.runtime.device.support_fp64:
    SUPPORTED_DTYPES.extend([torch.float64, torch.complex128])
    COMPLEX_DTYPES.append(torch.complex128)

DEVICE_REASON = "linalg_cross currently targets NVIDIA and Ascend"


def _randn(shape, dtype):
    if dtype.is_complex and flag_gems.vendor_name == "ascend":
        # torch_npu does not implement in-place normal generation for complex
        # tensors, so create the test values on CPU and transfer them instead.
        return torch.randn(shape, dtype=dtype).to(flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _assert_cross_close(result, reference, dtype):
    if flag_gems.vendor_name == "ascend" and dtype == torch.complex64:
        # aclnnIsClose does not accept complex64. Compare the real-valued view
        # on NPU so validation does not fall back to CPU.
        utils.gems_assert_close(
            torch.view_as_real(result),
            torch.view_as_real(reference),
            torch.float32,
        )
    elif dtype == torch.complex128:
        result = utils.to_cpu(result, reference)
        torch.testing.assert_close(result, reference, rtol=1e-10, atol=1e-10)
    else:
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.linalg_cross
@pytest.mark.skipif(not DEVICE_AVAILABLE, reason=DEVICE_REASON)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "input_shape,other_shape,dim",
    [
        ((2, 3, 4), (1, 3, 4), 1),
        ((4096, 3, 4), (1, 3, 4), 1),
        ((1, 3), (5, 3), -1),
        ((2, 4, 3), (2, 4, 3), -1),
    ],
)
def test_linalg_cross(input_shape, other_shape, dim, dtype):
    input = _randn(input_shape, dtype)
    other = _randn(other_shape, dtype)
    ref_input = utils.to_reference(input)
    ref_other = utils.to_reference(other)

    if flag_gems.vendor_name == "ascend":
        ref_out = torch.linalg.cross(input, other, dim=dim)
    else:
        ref_out = torch.linalg.cross(ref_input, ref_other, dim=dim)
    with flag_gems.use_gems(include=["linalg_cross"]):
        result = torch.linalg.cross(input, other, dim=dim)

    _assert_cross_close(result, ref_out, dtype)


@pytest.mark.linalg_cross
@pytest.mark.skipif(not DEVICE_AVAILABLE, reason=DEVICE_REASON)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_linalg_cross_noncontiguous_input_and_out(dtype):
    input = _randn((2, 4, 3), dtype).transpose(1, 2)
    other = _randn((1, 4, 3), dtype).transpose(1, 2)
    out = torch.empty((2, 4, 3), dtype=dtype, device=flag_gems.device).transpose(1, 2)
    ref_input = utils.to_reference(input)
    ref_other = utils.to_reference(other)
    if flag_gems.vendor_name == "ascend":
        ref_out = torch.empty(
            (2, 4, 3), dtype=dtype, device=input.device
        ).transpose(1, 2)
        torch.ops.aten.linalg_cross.out(input, other, dim=1, out=ref_out)
    else:
        ref_out = torch.empty(
            (2, 4, 3), dtype=dtype, device=ref_input.device
        ).transpose(1, 2)
        torch.ops.aten.linalg_cross.out(ref_input, ref_other, dim=1, out=ref_out)
    with flag_gems.use_gems(include=["linalg_cross_out"]):
        result = torch.ops.aten.linalg_cross.out(input, other, dim=1, out=out)

    assert result is out
    _assert_cross_close(out, ref_out, dtype)


@pytest.mark.linalg_cross
@pytest.mark.skipif(not DEVICE_AVAILABLE, reason=DEVICE_REASON)
def test_linalg_cross_rejects_different_input_ranks():
    input = _randn((3,), torch.float32)
    other = _randn((1, 3), torch.float32)

    with flag_gems.use_gems(include=["linalg_cross"]), pytest.raises(
        RuntimeError, match="same number of dimensions"
    ):
        torch.linalg.cross(input, other)


@pytest.mark.linalg_cross
@pytest.mark.skipif(not DEVICE_AVAILABLE, reason=DEVICE_REASON)
def test_linalg_cross_backward():
    input = _randn((2, 3, 4), torch.float32).requires_grad_()
    other = _randn((1, 3, 4), torch.float32).requires_grad_()
    if flag_gems.vendor_name == "ascend":
        ref_input = input.detach().clone().requires_grad_()
        ref_other = other.detach().clone().requires_grad_()
        ref_result = torch.linalg.cross(ref_input, ref_other, dim=1)
    else:
        ref_input = utils.to_reference(input.detach()).requires_grad_()
        ref_other = utils.to_reference(other.detach()).requires_grad_()
        ref_result = torch.linalg.cross(ref_input, ref_other, dim=1)

    ref_loss = ref_result.square().sum()
    ref_loss.backward()

    with flag_gems.use_gems(include=["linalg_cross"]):
        loss = torch.linalg.cross(input, other, dim=1).square().sum()
        loss.backward()

    utils.gems_assert_close(input.grad, ref_input.grad, torch.float32)
    utils.gems_assert_close(other.grad, ref_other.grad, torch.float32)


@pytest.mark.linalg_cross
@pytest.mark.skipif(not DEVICE_AVAILABLE, reason=DEVICE_REASON)
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_linalg_cross_conjugated_view(dtype):
    input = _randn((2, 3, 4), dtype).conj()
    other = _randn((1, 3, 4), dtype)
    ref_input = utils.to_reference(input)
    ref_other = utils.to_reference(other)

    if flag_gems.vendor_name == "ascend":
        ref_out = torch.linalg.cross(input, other, dim=1)
    else:
        ref_out = torch.linalg.cross(ref_input, ref_other, dim=1)
    with flag_gems.use_gems(include=["linalg_cross"]):
        result = torch.linalg.cross(input, other, dim=1)

    _assert_cross_close(result, ref_out, dtype)
