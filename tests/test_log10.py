import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


@pytest.mark.log10
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_out(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.log10(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_special_values(dtype):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, 10.0, -1.0, float("inf"), float("-inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_empty(dtype):
    shapes = ((0,), (4, 0), (2, 0, 3))
    for shape in shapes:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp, True)

        ref_out = torch.log10(ref_inp)
        with flag_gems.use_gems():
            res_out = torch.log10(inp)

        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_noncontiguous(dtype):
    inp = torch.rand((32, 64), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_log10_int_promotes_to_float(dtype):
    inp = torch.randint(1, 100, (128, 64), dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


# ── complex log10 tests ───────────────────────────────────────────────────────


@pytest.mark.log10
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_log10_complex(shape, dtype):
    if cfg.TO_CPU and dtype == torch.complex32:
        pytest.skip("complex32 not supported on CPU")

    inp = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10_
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_log10_inplace_complex(dtype):
    if cfg.TO_CPU and dtype == torch.complex32:
        pytest.skip("complex32 not supported on CPU")

    inp = torch.randn((64, 64), dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10_out
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_log10_out_complex(dtype):
    if cfg.TO_CPU and dtype == torch.complex32:
        pytest.skip("complex32 not supported on CPU")

    inp = torch.randn((64, 64), dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.log10(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10
def test_log10_complex_special_values():
    """Verify known identities: log10(1)=0, log10(10)=1, log10(0)=-inf+0j."""
    inp = torch.tensor(
        [0 + 0j, 1 + 0j, 10 + 0j, 0 + 1j, 1 + 1j, -1 + 0j],
        dtype=torch.complex64,
        device=flag_gems.device,
    )
    ref = torch.log10(inp.to("cpu").to(torch.complex128))
    with flag_gems.use_gems():
        res = torch.log10(inp)

    utils.gems_assert_close(res.to(torch.complex128), ref.to(flag_gems.device), torch.complex64, equal_nan=True)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_log10_complex_noncontiguous(dtype):
    if cfg.TO_CPU and dtype == torch.complex32:
        pytest.skip("complex32 not supported on CPU")

    base = torch.randn((32, 64), dtype=dtype, device="cpu").to(flag_gems.device)
    inp = base[::2, ::4]  # non-contiguous view
    ref_inp = utils.to_reference(inp.contiguous(), True)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out.contiguous(), ref_out, dtype, equal_nan=True)
