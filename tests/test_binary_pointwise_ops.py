import logging
import random

import numpy as np
import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    SCALARS,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import TO_CPU


def replace_zeros(inp):
    return torch.where(inp == 0, 1, inp)


@pytest.mark.add
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


@pytest.mark.add
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


@pytest.mark.add
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


@pytest.mark.add
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_accuracy_add_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random()))
        inp2 = float(np.float32(random.random()))
        alpha = float(np.float32(random.random()))
    else:
        inp1 = random.randint(0, 100)
        inp2 = random.randint(0, 100)
        alpha = random.randint(0, 100)

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    if dtype == torch.int64:
        gems_assert_equal(res_out, ref_out)
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.bitwise_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cuda")
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cuda")
    else:
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


@pytest.mark.bitwise_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_scalar(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cuda")
        inp2 = bool(random.randint(0, 2))
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
        )
        inp2 = 0x00FF
    ref_inp1 = to_reference(inp1)

    ref_out = torch.bitwise_and(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_scalar_tensor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = bool(random.randint(0, 2))
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cuda")
    else:
        inp1 = 0x00FF
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
        )
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_and(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.or_
@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cuda")
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cuda")
    else:
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


@pytest.mark.or_
@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cuda")
        inp2 = bool(random.randint(0, 2))
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
        )
        inp2 = 0x00FF
    ref_inp1 = to_reference(inp1)

    ref_out = torch.bitwise_or(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.or_
@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar_tensor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = bool(random.randint(0, 2))
        inp2 = torch.randint(0, 2, size=shape, dtype=torch.bool, device="cuda")
    else:
        inp1 = 0x00FF
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
        )
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_or(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.clamp
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


@pytest.mark.clamp
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


@pytest.mark.div
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar
    ref_inp1 = to_reference(inp1, False)

    ref_out = torch.div(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2, False)

    ref_out = torch.div(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_accuracy_div_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random() + 0.01))
        inp2 = float(np.float32(random.random() + 0.01))
    else:
        inp1 = random.randint(1, 100)
        inp2 = random.randint(1, 100)

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    if dtype == torch.int64:
        gems_assert_equal(res_out, ref_out)
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.trunc_divide
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
# Note : tl.math.div_rz only support float32, cast will cause diff
# with torch, so we only do float32 test for now.
def test_accuracy_trunc_div(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode="trunc")

    if not TO_CPU:
        logging.debug(
            f"The maximum difference between torch and triton is "
            f"{torch.max(torch.abs(ref_out - res_out))}"
        )
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.trunc_divide
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_accuracy_trunc_divide_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random() + 0.01))
        inp2 = float(np.float32(random.random() + 0.01))
    else:
        inp1 = random.randint(1, 100)
        inp2 = random.randint(1, 100)

    ref_out = torch.div(inp1, inp2, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode="trunc")

    if dtype == torch.int64:
        gems_assert_equal(res_out, ref_out)
    else:
        gems_assert_close(res_out, ref_out, dtype)


# TODO: failed at large size, eg. (65536 * 2048,)
@pytest.mark.floor_divide
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_floor_div_float(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2, rounding_mode="floor")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode="floor")

    gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.floor_divide
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_floor_div_int(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cuda",
    )
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cuda",
    )
    if TO_CPU:
        inp1 = replace_zeros(inp1)
        inp2 = replace_zeros(inp2)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = ref_inp1 // ref_inp2
    with flag_gems.use_gems():
        res_out = inp1 // inp2

    gems_assert_equal(res_out, ref_out)

    for d in inp2.flatten()[:100]:
        ref_d = to_reference(d, False)
        ref_out = ref_inp1 // ref_d
        with flag_gems.use_gems():
            res_out = inp1 // d
        gems_assert_equal(res_out, ref_out)

        ref_out = ref_d // ref_inp1
        with flag_gems.use_gems():
            res_out = d // inp1
        gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_divide
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_accuracy_floor_divide_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random() + 0.01))
        inp2 = float(np.float32(random.random() + 0.01))
    else:
        inp1 = random.randint(1, 100)
        inp2 = random.randint(1, 100)

    ref_out = torch.floor_divide(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.floor_divide(inp1, inp2)

    if dtype == torch.int64:
        gems_assert_equal(res_out, ref_out)
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.remainder
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_remainder(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cuda",
    )
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cuda",
    )
    if TO_CPU:
        inp1 = replace_zeros(inp1)
        inp2 = replace_zeros(inp2)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = ref_inp1 % ref_inp2
    with flag_gems.use_gems():
        res_out = inp1 % inp2

    gems_assert_equal(res_out, ref_out)

    for d in inp2.flatten()[:100]:
        ref_d = to_reference(d, False)
        ref_out = ref_inp1 % ref_d
        with flag_gems.use_gems():
            res_out = inp1 % d
        gems_assert_equal(res_out, ref_out)

        ref_out = ref_d % ref_inp1
        with flag_gems.use_gems():
            res_out = d % inp1
        gems_assert_equal(res_out, ref_out)


@pytest.mark.eq
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


@pytest.mark.eq
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


@pytest.mark.ge
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


@pytest.mark.ge
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


@pytest.mark.gelu_and_mul
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


@pytest.mark.gt
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


@pytest.mark.gt
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


@pytest.mark.le
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


@pytest.mark.le
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


@pytest.mark.lt
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


@pytest.mark.lt
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


@pytest.mark.mul
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


@pytest.mark.mul
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


@pytest.mark.mul
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


@pytest.mark.mul
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_accuracy_mul_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random()))
        inp2 = float(np.float32(random.random()))
    else:
        inp1 = random.randint(0, 100)
        inp2 = random.randint(0, 100)

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    if dtype == torch.int64:
        gems_assert_equal(res_out, ref_out)
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ne
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


@pytest.mark.ne
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


@pytest.mark.pow
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


@pytest.mark.maximum
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_maximum(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.maximum(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.maximum(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.minimum
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_minimum(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.minimum(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.minimum(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.pow
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


@pytest.mark.pow
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


@pytest.mark.rsub
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


@pytest.mark.silu_and_mul
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


@pytest.mark.sub
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


@pytest.mark.sub
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


@pytest.mark.sub
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


@pytest.mark.sub
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_accuracy_sub_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random()))
        inp2 = float(np.float32(random.random()))
        alpha = float(np.float32(random.random()))
    else:
        inp1 = random.randint(0, 100)
        inp2 = random.randint(0, 100)
        alpha = random.randint(0, 100)

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    if dtype == torch.int64:
        gems_assert_equal(res_out, ref_out)
    else:
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.where
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_self_out_cross_device(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")

    import itertools

    shapes = (shape, None)
    for a_shape, b_shape, c_shape in itertools.product(shapes, shapes, shapes):
        a = inp1 if a_shape else torch.tensor(0)
        b = inp2 if b_shape else torch.tensor(1)
        c = cond if c_shape else torch.tensor(True)

        ref_a = to_reference(a)
        ref_b = to_reference(b)
        ref_c = to_reference(c)

        ref_out = torch.where(ref_c, ref_a, ref_b)
        with flag_gems.use_gems():
            res_out = torch.where(c, a, b)

        gems_assert_equal(res_out, ref_out)


@pytest.mark.where
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_self_out(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
    out = torch.empty(shape, dtype=dtype, device="cuda")
    ref_out = to_reference(out)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)
    ref_cond = to_reference(cond)

    ref_out = torch.where(ref_cond, ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.where(cond, inp1, inp2, out=out)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.where
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


@pytest.mark.where
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_scalar_self(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2)

    ref_out = torch.where(ref_inp2 > 0, inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.where(inp2 > 0, inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.where
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_where_scalar_other(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp2 = to_reference(inp2)

    ref_out = torch.where(ref_inp2 > 0, ref_inp2, inp1)
    with flag_gems.use_gems():
        res_out = torch.where(inp2 > 0, inp2, inp1)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.isclose
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
@pytest.mark.parametrize("zero_tol", [False, True])
@pytest.mark.parametrize("equal_nan", [False, True])
@pytest.mark.parametrize("gen_nan", [0, 1, 2, 3, 4])
def test_accuracy_isclose(shape, dtype, zero_tol, equal_nan, gen_nan):
    # [gen_nan] 1: nan, 2: inf, 3: -inf, 4: inf vs -inf
    rtol = (
        torch.rand(1, dtype=torch.float32, device="cuda").item() * 0.0001
        if not zero_tol
        else 0
    )
    if dtype in ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device="cuda")
        inp2 = torch.randn(shape, dtype=dtype, device="cuda")
        if gen_nan:
            nan_num = torch.full(
                (1,),
                float("nan" if gen_nan == 1 else "inf"),
                dtype=dtype,
                device="cuda",
            )
            inp1.view(-1)[0] = -nan_num if gen_nan == 3 else nan_num
            inp2.view(-1)[0] = -nan_num if gen_nan >= 3 else nan_num
        atol = (
            torch.finfo(dtype).tiny * torch.randint(0, 4, (1,), device="cuda").item()
            if not zero_tol
            else 0
        )
    else:
        inp1 = torch.randint(-1000, 1000, shape, device="cuda").to(dtype)
        inp2 = torch.randint(-1000, 1000, shape, device="cuda").to(dtype)
        if dtype in [torch.int64]:
            inp1.view(-1)[0] = 2**63 - 1
            inp2.view(-1)[0] = -(2**63)
            if inp1.numel() > 2 and inp2.numel() > 2:
                inp1.view(-1)[1] = 2**60 + 2**20
                inp2.view(-1)[1] = 2**60
                inp1.view(-1)[2] = 2**60 + 1
                inp2.view(-1)[2] = 2**60
            atol = 2 if not zero_tol else 0
            if gen_nan == 0:
                rtol = 0
        elif dtype in [torch.int32]:
            inp1.view(-1)[0] = 2**31 - 1
            inp2.view(-1)[0] = -(2**31)
            if inp1.numel() > 2 and inp2.numel() > 2:
                inp1.view(-1)[1] = 2**30 + 2**5
                inp2.view(-1)[1] = 2**30
                inp1.view(-1)[2] = 2**30 + 1
                inp2.view(-1)[2] = 2**30
            atol = 2 if not zero_tol else 0
            if gen_nan == 0:
                rtol = 0
        else:
            atol = (
                (
                    torch.finfo(torch.float16).eps
                    * torch.randint(0, 10, (1,), device="cuda").item()
                )
                if not zero_tol
                else 0
            )

    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)
    logging.debug(
        "shape={}, dtype={}, rtol={}, atol={}".format(shape, dtype, rtol, atol)
    )

    with flag_gems.use_gems():
        res_out = torch.isclose(inp1, inp2, rtol, atol, equal_nan=equal_nan)
    ref_out = torch.isclose(ref_inp1, ref_inp2, rtol, atol, equal_nan=equal_nan)

    inp1_flat = inp1.view(-1)
    inp2_flat = inp2.view(-1)
    ref_flat = ref_out.view(-1)
    res_flat = res_out.view(-1)
    if dtype in FLOAT_DTYPES and gen_nan:
        logging.debug(
            "equal_nan={}, gen_nan={}: inp1={}, inp2={}, res={}, ref={}".format(
                equal_nan,
                gen_nan,
                inp1_flat[0],
                inp2_flat[0],
                res_flat[0],
                ref_flat[0],
            )
        )
    if inp1.numel() > 2 and dtype in [torch.int64, torch.int32]:
        assert (
            res_flat[1] == ref_flat[1] and res_flat[2] == ref_flat[2]
        ), "res vs ref: {} vs {}, {} vs {}".format(
            res_flat[1], ref_flat[1], res_flat[2], ref_flat[2]
        )
    gems_assert_equal(res_out, ref_out)


@pytest.mark.allclose
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
@pytest.mark.parametrize("equal_nan", [False, True])
@pytest.mark.parametrize("gen_nan", [0, 1, 2, 3, 4])
def test_accuracy_allclose(shape, dtype, equal_nan, gen_nan):
    # [gen_nan] 1: nan, 2: inf, 3: -inf, 4: inf vs -inf
    rtol = torch.rand(1, dtype=torch.float32, device="cuda").item() * (
        0.0001 if dtype in [torch.bfloat16, torch.float16] else 0.01
    )
    if dtype in ALL_FLOAT_DTYPES:
        atol = torch.finfo(dtype).tiny * torch.randint(0, 4, (1,), device="cuda").item()
        inp1 = torch.full(shape, 1.234, dtype=dtype, device="cuda")
        inp2 = torch.full(shape, 1.234, dtype=dtype, device="cuda")
        if gen_nan:
            nan_num = torch.full(
                (1,),
                float("nan" if gen_nan == 1 else "inf"),
                dtype=dtype,
                device="cuda",
            )
            inp1.view(-1)[0] = -nan_num if gen_nan == 3 else nan_num
            inp2.view(-1)[0] = -nan_num if gen_nan >= 3 else nan_num
    else:
        atol = (
            torch.finfo(torch.float16).eps
            * torch.randint(0, 10, (1,), device="cuda").item()
        )
        inp1 = torch.randint(-1000, 1000, shape, device="cuda").to(dtype)
        inp2 = torch.randint(-1000, 1000, shape, device="cuda").to(dtype)

    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)
    logging.debug(
        "shape={}, dtype={}, rtol={}, atol={}".format(shape, dtype, rtol, atol)
    )

    with flag_gems.use_gems():
        res_out = torch.allclose(inp1, inp2, rtol, atol, equal_nan=equal_nan)
    ref_out = torch.allclose(ref_inp1, ref_inp2, rtol, atol, equal_nan=equal_nan)

    assert res_out == ref_out


@pytest.mark.logical_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES + BOOL_TYPES)
def test_accuracy_logical_or(shape, dtype):
    if dtype in ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device="cuda")
        inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cuda")
        inp2 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cuda")
    elif dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
        inp2 = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.logical_or(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.logical_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.logical_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES + BOOL_TYPES)
def test_accuracy_logical_and(shape, dtype):
    if dtype in ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device="cuda")
        inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cuda")
        inp2 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cuda")
    elif dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
        inp2 = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.logical_and(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.logical_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.logical_xor
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES + BOOL_TYPES)
def test_accuracy_logical_xor(shape, dtype):
    if dtype in ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device="cuda")
        inp2 = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cuda")
        inp2 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cuda")
    elif dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
        inp2 = torch.randint(0, 2, shape, dtype=dtype, device="cuda")
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.logical_xor(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.logical_xor(inp1, inp2)

    gems_assert_equal(res_out, ref_out)
