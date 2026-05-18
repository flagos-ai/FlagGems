import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


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


@pytest.mark.log10
@pytest.mark.parametrize("shape", [(), (1,), (2,), (17,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_tiny_shapes(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.25

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_
@pytest.mark.parametrize("shape", [(), (1,), (2,), (17,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_inplace_tiny_shapes(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.25
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10_out
@pytest.mark.parametrize("shape", [(), (1,), (2,), (17,)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log10_out_tiny_shapes(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.25
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.log10(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log10
def test_log10_complex64():
    inp = torch.tensor(
        [1.0 + 0.0j, 10.0 + 1.0j, -1.0 + 0.5j, 0.25 - 2.0j],
        dtype=torch.complex64,
        device=flag_gems.device,
    )

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, torch.complex64)


@pytest.mark.log10_
def test_log10_complex64_inplace():
    inp = torch.tensor(
        [1.0 + 0.0j, 10.0 + 1.0j, -1.0 + 0.5j, 0.25 - 2.0j],
        dtype=torch.complex64,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)

    utils.gems_assert_close(res_out, ref_out, torch.complex64)


@pytest.mark.log10_out
def test_log10_complex64_out():
    inp = torch.tensor(
        [1.0 + 0.0j, 10.0 + 1.0j, -1.0 + 0.5j, 0.25 - 2.0j],
        dtype=torch.complex64,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.log10(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, torch.complex64)


@pytest.mark.log10
def test_log10_complex64_special_values():
    inp = torch.tensor(
        [
            0.0 + 0.0j,
            1.0 + 0.0j,
            -1.0 + 0.0j,
            10.0 + 1.0j,
            complex(float("inf"), 1.0),
            complex(float("inf"), float("inf")),
            complex(float("-inf"), float("inf")),
            complex(float("inf"), float("-inf")),
            complex(float("-inf"), float("-inf")),
            complex(float("nan"), 1.0),
        ],
        dtype=torch.complex64,
        device=flag_gems.device,
    )

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, torch.complex64, equal_nan=True)


@pytest.mark.log10
def test_log10_complex64_noncontiguous():
    inp = torch.randn(
        (8, 16), dtype=torch.complex64, device=flag_gems.device
    ).transpose(0, 1)

    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    utils.gems_assert_close(res_out, ref_out, torch.complex64)


@pytest.mark.log10
def test_log10_complex64_out_dtype_semantics():
    inp = torch.tensor(
        [1.0 + 0.0j, 10.0 + 1.0j, -1.0 + 0.5j],
        dtype=torch.complex64,
        device=flag_gems.device,
    )

    for out_dtype in (torch.complex64, torch.complex128):
        ref_out = torch.empty(inp.shape, dtype=out_dtype, device=flag_gems.device)
        torch.log10(inp, out=ref_out)
        with flag_gems.use_gems():
            res_out = torch.empty(inp.shape, dtype=out_dtype, device=flag_gems.device)
            torch.log10(inp, out=res_out)

        torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1.3e-6)

    with flag_gems.use_gems():
        out = torch.empty(inp.shape, dtype=torch.float32, device=flag_gems.device)
        with pytest.raises(RuntimeError, match="can't be cast"):
            torch.log10(inp, out=out)


@pytest.mark.log10
def test_log10_complex64_signed_zero_branch_cut():
    inp = torch.tensor(
        [-0.0 + 0.0j, 0.0 - 0.0j, -1.0 - 0.0j, -1.0 + 0.0j],
        dtype=torch.complex64,
        device=flag_gems.device,
    )

    ref_out = torch.log10(inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1.3e-6)
    torch.testing.assert_close(
        torch.signbit(torch.view_as_real(res_out)),
        torch.signbit(torch.view_as_real(ref_out)),
    )


@pytest.mark.log10
@pytest.mark.skipif(
    not utils.fp64_is_supported, reason="complex128 log10 requires fp64 support"
)
def test_log10_complex128():
    inp = torch.tensor(
        [1.0 + 0.0j, 10.0 + 1.0j, -1.0 + 0.5j, 0.25 - 2.0j],
        dtype=torch.complex128,
        device=flag_gems.device,
    )

    ref_out = torch.log10(inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    torch.testing.assert_close(res_out, ref_out, rtol=1e-7, atol=1e-7)
