import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

LOGDET_SHAPES = (
    [(2, 2), (16, 16), (64, 64)]
    if QUICK_MODE
    else [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
    ]
)
LOGDET_BATCH_SHAPES = (
    [(2, 3, 8, 8)]
    if QUICK_MODE
    else [(2, 3, 3), (4, 8, 8), (2, 3, 16, 16), (128, 8, 8)]
)
REAL_DTYPES = [torch.float32] + ([torch.float64] if utils.fp64_is_supported else [])
COMPLEX_DTYPES = [torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)


def _positive_definite(shape, dtype):
    n = shape[-1]
    matrix = torch.randn(shape, dtype=dtype, device=flag_gems.device) / math.sqrt(n)
    eye = torch.eye(n, dtype=dtype, device=flag_gems.device)
    return matrix @ matrix.transpose(-2, -1) + eye


def _assert_logdet_close(result, reference, dtype, n):
    utils.gems_assert_close(result, reference, dtype, reduce_dim=n)


@pytest.mark.logdet
@pytest.mark.parametrize("shape", LOGDET_SHAPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_logdet_positive_definite(shape, dtype):
    inp = _positive_definite(shape, dtype)
    ref_inp = utils.to_reference(inp)
    reference = torch.logdet(ref_inp)

    result = flag_gems.logdet(inp)

    _assert_logdet_close(result, reference, dtype, shape[-1])


@pytest.mark.logdet
@pytest.mark.parametrize("shape", LOGDET_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_logdet_batch(shape, dtype):
    inp = _positive_definite(shape, dtype)
    ref_inp = utils.to_reference(inp)
    reference = torch.logdet(ref_inp)

    result = flag_gems.logdet(inp)

    _assert_logdet_close(result, reference, dtype, shape[-1])


@pytest.mark.logdet
@pytest.mark.skipif(not utils.fp64_is_supported, reason="FP64 is not supported")
def test_logdet_fp64_large_batch_4x4():
    inp = _positive_definite((4096, 4, 4), torch.float64)
    scales = torch.where(
        torch.arange(4096, device=flag_gems.device) % 2 == 0,
        torch.tensor(1e100, dtype=torch.float64, device=flag_gems.device),
        torch.tensor(1e-100, dtype=torch.float64, device=flag_gems.device),
    )
    inp *= scales[:, None, None]
    ref_inp = utils.to_reference(inp)
    reference = torch.logdet(ref_inp)

    result = flag_gems.logdet(inp)

    _assert_logdet_close(result, reference, torch.float64, 4)


@pytest.mark.logdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_logdet_negative_and_singular(dtype):
    positive = _positive_definite((3, 4, 4), dtype)
    negative = positive.clone()
    negative[..., [0, 1], :] = negative[..., [1, 0], :]
    singular = positive.clone()
    singular[..., 0, :] = 0
    inp = torch.cat((negative, singular), dim=0)
    ref_inp = utils.to_reference(inp)
    reference = torch.logdet(ref_inp)

    result = flag_gems.logdet(inp)

    utils.gems_assert_equal(result, reference, equal_nan=True)


@pytest.mark.logdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_logdet_tiny_nonzero_pivots(dtype):
    tiny = 1e-20 if dtype == torch.float32 else 1e-200
    diagonal = torch.tensor([tiny, 2.0, 3.0], dtype=dtype, device=flag_gems.device)
    inp = torch.diag(diagonal)
    ref_inp = utils.to_reference(inp)
    reference = torch.logdet(ref_inp)

    result = flag_gems.logdet(inp)

    _assert_logdet_close(result, reference, dtype, 3)
    assert torch.isfinite(result)


@pytest.mark.logdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_logdet_noncontiguous_and_exported_api(dtype):
    base = _positive_definite((3, 7, 7), dtype)
    inp = base.transpose(-2, -1)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)
    reference = torch.logdet(ref_inp)

    result = flag_gems.logdet(inp)

    _assert_logdet_close(result, reference, dtype, 7)


@pytest.mark.logdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_logdet_empty(dtype):
    for shape in ((0, 0), (3, 0, 0), (0, 4, 4)):
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)
        reference = torch.logdet(ref_inp)

        result = flag_gems.logdet(inp)

        utils.gems_assert_equal(result, reference)


@pytest.mark.logdet
@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_logdet_complex_fallback(dtype):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn((2, 4, 4), dtype=real_dtype, device=flag_gems.device)
    imag = torch.randn_like(real)
    inp = torch.complex(real, imag) + 4 * torch.eye(
        4, dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)
    reference = torch.logdet(ref_inp)

    result = flag_gems.logdet(inp)

    _assert_logdet_close(result, reference, dtype, 4)


@pytest.mark.logdet
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_logdet_autograd_fallback(dtype):
    inp = _positive_definite((4, 4), dtype).detach().requires_grad_(True)
    ref_inp = utils.to_reference(inp).detach().requires_grad_(True)
    reference = torch.logdet(ref_inp)
    ref_grad = torch.autograd.grad(reference, ref_inp)[0]

    result = flag_gems.logdet(inp)
    result_grad = torch.autograd.grad(result, inp)[0]

    _assert_logdet_close(result, reference, dtype, 4)
    _assert_logdet_close(result_grad, ref_grad, dtype, 4)


@pytest.mark.logdet
def test_logdet_errors():
    invalid_inputs = [
        torch.randn((3,), device=flag_gems.device),
        torch.randn((3, 4), device=flag_gems.device),
        torch.ones((3, 3), dtype=torch.int32, device=flag_gems.device),
        torch.randn((3, 3), dtype=torch.float16, device=flag_gems.device),
    ]
    for inp in invalid_inputs:
        with pytest.raises(RuntimeError):
            flag_gems.logdet(inp)
