import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, INT_DTYPES, gems_assert_close, to_reference

LOG10_SHAPES = [
    (),
    (1,),
    (8, 8),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


@pytest.mark.log10
@pytest.mark.parametrize("shape", LOG10_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_accuracy_log10(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )

    ref_inp = to_reference(inp)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    if dtype in INT_DTYPES:
        gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)
    else:
        gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.log10_
@pytest.mark.parametrize("shape", LOG10_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone())

    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10_out
@pytest.mark.parametrize("shape", LOG10_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log10_out(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.log10(inp, out=res_out)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_edge_cases_log10(dtype):
    inp = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0],
        device=flag_gems.device,
        dtype=dtype,
    )
    ref_inp = to_reference(inp)

    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.log10_
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_edge_cases_log10_(dtype):
    inp = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0],
        device=flag_gems.device,
        dtype=dtype,
    )
    ref_inp = to_reference(inp.clone())

    ref_out = torch.log10_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10_(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10_out
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_edge_cases_log10_out(dtype):
    inp = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0],
        device=flag_gems.device,
        dtype=dtype,
    )
    ref_inp = to_reference(inp)

    ref_out = torch.empty_like(ref_inp)
    torch.log10(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.log10(inp, out=res_out)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log10_
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_type_error_log10_(dtype):
    inp = torch.randint(0, 1000, (1024,), device=flag_gems.device, dtype=dtype)

    with flag_gems.use_gems():
        with pytest.raises(RuntimeError, match="log_ only supports floating-point"):
            torch.log10_(inp)


@pytest.mark.log10_out
@pytest.mark.type
@pytest.mark.parametrize("inp_type", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("out_type", INT_DTYPES)
def test_type_error_log10_out(inp_type, out_type):
    if inp_type in FLOAT_DTYPES:
        inp = torch.randn((1024,), device=flag_gems.device, dtype=inp_type)
    else:
        inp = torch.randint(0, 1000, (1024,), device=flag_gems.device, dtype=inp_type)
    res_out = torch.empty_like(inp, dtype=out_type)

    with flag_gems.use_gems():
        with pytest.raises(
            RuntimeError,
            match="result type Float can't be cast to the desired output type",
        ):
            torch.log10(inp, out=res_out)
