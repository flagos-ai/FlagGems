import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SUPPORTED_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.complex64,
]
COMPLEX_DTYPES = [torch.complex64]
if flag_gems.runtime.device.support_fp64:
    SUPPORTED_DTYPES.extend([torch.float64, torch.complex128])
    COMPLEX_DTYPES.append(torch.complex128)


def _randn(shape, dtype):
    if dtype.is_complex and flag_gems.vendor_name == "ascend":
        # torch_npu does not implement in-place normal generation for complex
        # tensors, so create the test values on CPU and transfer them instead.
        return torch.randn(shape, dtype=dtype).to(flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _to_cross_reference(tensor, dtype):
    reference = utils.to_reference(tensor)
    if dtype in (torch.float16, torch.bfloat16):
        # CPU and accelerator linalg_cross implementations use different
        # low-precision intermediate rounding on some backends. FP32 opmath is
        # the common, more accurate reference for both execution locations.
        reference = reference.to(torch.float32)
    return reference


def _assert_cross_close(result, reference, dtype):
    if dtype == torch.complex64:
        # Some backends do not implement isclose/abs for complex tensors.
        # Compare the real-valued view without changing the reference device.
        utils.gems_assert_close(
            torch.view_as_real(result),
            torch.view_as_real(reference),
            torch.float32,
        )
    elif dtype == torch.complex128:
        result = utils.to_cpu(result, reference)
        torch.testing.assert_close(
            torch.view_as_real(result),
            torch.view_as_real(reference),
            rtol=1e-10,
            atol=1e-10,
        )
    elif dtype == torch.float16:
        utils.gems_assert_close(result, reference, dtype, atol=1e-3)
    elif dtype == torch.bfloat16:
        # BF16 has a wider quantization step than FP16 for unit-scale values.
        utils.gems_assert_close(result, reference, dtype, atol=1e-2)
    else:
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.linalg_cross
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "input_shape,other_shape,dim",
    [
        ((3, 4), (3, 4), 0),
        ((2, 3, 4), (1, 3, 4), 1),
        ((4096, 3, 4), (1, 3, 4), 1),
        ((2, 4, 3), (1, 4, 3), 2),
        ((2, 3, 4), (1, 3, 4), -2),
        ((1, 3), (5, 3), -1),
        ((2, 4, 3), (2, 4, 3), -1),
        ((2, 4, 3, 5), (1, 4, 3, 5), 2),
    ],
)
def test_linalg_cross(input_shape, other_shape, dim, dtype):
    input = _randn(input_shape, dtype)
    other = _randn(other_shape, dtype)
    ref_input = _to_cross_reference(input, dtype)
    ref_other = _to_cross_reference(other, dtype)

    ref_out = torch.linalg.cross(ref_input, ref_other, dim=dim)
    with flag_gems.use_gems(include=["linalg_cross"]):
        result = torch.linalg.cross(input, other, dim=dim)

    _assert_cross_close(result, ref_out, dtype)


@pytest.mark.linalg_cross_out
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_linalg_cross_noncontiguous_input_and_out(dtype):
    input = _randn((2, 4, 3), dtype).transpose(1, 2)
    other = _randn((1, 4, 3), dtype).transpose(1, 2)
    out = torch.empty((2, 4, 3), dtype=dtype, device=flag_gems.device).transpose(1, 2)
    ref_input = _to_cross_reference(input, dtype)
    ref_other = _to_cross_reference(other, dtype)
    ref_out = torch.empty(
        (2, 4, 3), dtype=ref_input.dtype, device=ref_input.device
    ).transpose(1, 2)
    torch.ops.aten.linalg_cross.out(ref_input, ref_other, dim=1, out=ref_out)
    with flag_gems.use_gems(include=["linalg_cross_out"]):
        result = torch.ops.aten.linalg_cross.out(input, other, dim=1, out=out)

    assert result is out
    _assert_cross_close(out, ref_out, dtype)


@pytest.mark.linalg_cross
def test_linalg_cross_rejects_different_input_ranks():
    input = _randn((3,), torch.float32)
    other = _randn((1, 3), torch.float32)

    with (
        flag_gems.use_gems(include=["linalg_cross"]),
        pytest.raises(RuntimeError, match="same number of dimensions"),
    ):
        torch.linalg.cross(input, other)


@pytest.mark.linalg_cross
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_linalg_cross_conjugated_view(dtype):
    input = _randn((2, 3, 4), dtype).conj()
    other = _randn((1, 3, 4), dtype)
    ref_input = utils.to_reference(input)
    ref_other = utils.to_reference(other)

    ref_out = torch.linalg.cross(ref_input, ref_other, dim=1)
    with flag_gems.use_gems(include=["linalg_cross"]):
        result = torch.linalg.cross(input, other, dim=1)

    _assert_cross_close(result, ref_out, dtype)
