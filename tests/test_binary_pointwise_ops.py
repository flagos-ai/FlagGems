import pytest
import torch

import flag_gems

from .accuracy_utils import *


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.add(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.add(ref_inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.add(inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseand(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_and(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseand_scalar(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    inp2 = 0x00FF
    ref_inp1 = to_reference(inp1)

    ref_out = torch.bitwise_and(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseand_scalar_tensor(shape, dtype):
    inp1 = 0x00FF
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_and(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseor(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_or(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseor_scalar(shape, dtype):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    inp2 = 0x00FF
    ref_inp1 = to_reference(inp1)

    ref_out = torch.bitwise_or(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_bitwiseor_scalar_tensor(shape, dtype):
    inp1 = 0x00FF
    inp2 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
    )
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_or(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("maxi", SCALARS)
@pytest.mark.parametrize("mini", SCALARS)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp(shape, maxi, mini, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = to_reference(inp)

    ref_out = torch.clamp(ref_inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp(inp, min=mini, max=maxi)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp_tensor(shape, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    maxi = torch.randn(shape, dtype=dtype, device="cuda")
    mini = torch.randn(shape, dtype=dtype, device="cuda")
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = to_reference(inp)
    ref_maxi = to_reference(maxi)
    ref_mini = to_reference(mini)

    ref_out = torch.clamp(ref_inp, min=ref_mini, max=ref_maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp(inp, min=mini, max=maxi)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.div(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.div(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_eq(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device="cuda")
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.eq(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.eq(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_eq_scalar(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device="cuda")
    inp2 = 0
    ref_inp1 = to_reference(inp1)

    ref_out = torch.eq(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.eq(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ge(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.ge(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.ge(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ge_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = 0
    ref_inp1 = to_reference(inp1)

    ref_out = torch.ge(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.ge(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu_and_mul(shape, approximate, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(
        torch.nn.functional.gelu(ref_inp1, approximate=approximate), ref_inp2
    )
    with flag_gems.use_gems():
        res_out = flag_gems.gelu_and_mul(inp1, inp2, approximate)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gt(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.gt(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.gt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gt_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    inp2 = 0

    ref_out = torch.gt(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.gt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_le(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.le(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.le(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_le_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = 0
    ref_inp1 = to_reference(inp1)

    ref_out = torch.le(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.le(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_lt(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.lt(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.lt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_lt_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = 0
    ref_inp1 = to_reference(inp1)

    ref_out = torch.lt(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.lt(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.mul(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ne(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device="cuda")
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.ne(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.ne(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_ne_scalar(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device="cuda")
    inp2 = 0
    ref_inp1 = to_reference(inp1)

    ref_out = torch.ne(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.ne(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.pow(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_scalar_tensor(scalar, shape, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.pow(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_tensor_scalar(scalar, shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.pow(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.rsub(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.rsub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu_and_mul(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(torch.nn.functional.silu(ref_inp1), ref_inp2)
    with flag_gems.use_gems():
        res_out = flag_gems.silu_and_mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.sub(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.sub(ref_inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.sub(inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_self(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.where(ref_inp1 > 0, ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.where(inp1 > 0, inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_scalar_self(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2)

    ref_out = torch.where(inp2 > 0, inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.where(inp2 > 0, inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_scalar_other(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2)

    ref_out = torch.where(inp2 > 0, ref_inp2, inp1)
    with flag_gems.use_gems():
        res_out = torch.where(inp2 > 0, inp2, inp1)

    gems_assert_equal(res_out, ref_out)
