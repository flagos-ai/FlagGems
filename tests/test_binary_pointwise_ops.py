import logging
import math
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.add(ref_inp1, ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.add_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.clone(), True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = ref_inp1.add_(ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.add_(inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.add(ref_inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.add_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_tensor_scalar_(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = to_reference(inp1.clone(), True)

    ref_out = ref_inp1.add_(inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.add_(inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_and(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_and_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp1 = to_reference(inp1.clone())
    ref_inp2 = to_reference(inp2)

    ref_out = ref_inp1.bitwise_and_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.bitwise_and_(inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_scalar(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = bool(random.randint(0, 2))
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = 0x00FF
    ref_inp1 = to_reference(inp1)

    ref_out = torch.bitwise_and(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_and_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_scalar_(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
        inp2 = bool(random.randint(0, 2))
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = 0x00FF
    ref_inp1 = to_reference(inp1.clone())

    ref_out = ref_inp1.bitwise_and_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.bitwise_and_(inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_and
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseand_scalar_tensor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = bool(random.randint(0, 2))
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    else:
        inp1 = 0x00FF
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_and(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.bitwise_or(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_or_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp1 = to_reference(inp1.clone())
    ref_inp2 = to_reference(inp2)

    ref_out = ref_inp1.bitwise_or_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.bitwise_or_(inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = bool(random.randint(0, 2))
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = 0x00FF
    ref_inp1 = to_reference(inp1)

    ref_out = torch.bitwise_or(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_or(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_or_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar_(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
        inp2 = bool(random.randint(0, 2))
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = 0x00FF
    ref_inp1 = to_reference(inp1.clone())

    ref_out = ref_inp1.bitwise_or_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.bitwise_or_(inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwiseor_scalar_tensor(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = bool(random.randint(0, 2))
        inp2 = torch.randint(0, 2, size=shape, dtype=torch.bool, device="cpu").to(
            flag_gems.device
        )
    else:
        inp1 = 0x00FF
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
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
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = to_reference(inp)

    ref_out = torch.clamp(ref_inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp(inp, min=mini, max=maxi)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.clamp_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("maxi", SCALARS)
@pytest.mark.parametrize("mini", SCALARS)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp_(shape, maxi, mini, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = to_reference(inp.clone())

    ref_out = torch.clamp_(ref_inp, min=mini, max=maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp_(inp, min=mini, max=maxi)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.clamp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp_tensor(shape, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    maxi = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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


@pytest.mark.inplace
@pytest.mark.clamp_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("isnone", [None, "max", "min"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp_tensor_(shape, isnone, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    maxi = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if isnone == "min":
        mini = None
    elif isnone == "max":
        maxi = None
    ref_inp = to_reference(inp.clone())
    ref_maxi = to_reference(maxi)
    ref_mini = to_reference(mini)

    ref_out = torch.clamp_(ref_inp, min=ref_mini, max=ref_maxi)
    with flag_gems.use_gems():
        res_out = torch.clamp_(inp, min=mini, max=maxi)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.clamp_min
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp_min(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_mini = to_reference(mini)

    ref_out = torch.clamp_min(ref_inp, min=ref_mini)
    with flag_gems.use_gems():
        res_out = torch.clamp_min(inp, min=mini)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.clamp_min_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_clamp_min_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mini = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone())
    ref_mini = to_reference(mini)

    ref_out = torch.clamp_min_(ref_inp, min=ref_mini)
    with flag_gems.use_gems():
        res_out = torch.clamp_min_(inp, min=mini)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.div
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.div_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_tensor_tensor_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.clone(), False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = to_reference(inp1, False)

    ref_out = torch.div(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.div_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_tensor_scalar_(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = to_reference(inp1.clone(), False)

    ref_out = ref_inp1.div_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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


@pytest.mark.div
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
# Note : tl.math.div_rz only support float32, cast will cause diff
# with torch, so we only do float32 test for now.
def test_accuracy_trunc_div(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1 = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)

    upcast = (
        True
        if flag_gems.vendor_name not in ["cambricon", "iluvatar", "kunlunxin"]
        else False
    )
    ref_inp1 = to_reference(inp1, upcast)
    ref_inp2 = to_reference(inp2, upcast)

    ref_out = torch.div(ref_inp1, ref_inp2, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode="trunc")

    if not TO_CPU:
        logging.debug(
            f"The maximum difference between torch and triton is "
            f"{torch.max(torch.abs(ref_out - res_out))}"
        )
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.div_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
# Note : tl.math.div_rz only support float32, cast will cause diff
# with torch, so we only do float32 test for now.
def test_accuracy_trunc_div_(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    inp1 = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    if flag_gems.vendor_name in ("cambricon", "kunlunxin", "iluvatar"):
        upcast = False
    else:
        upcast = True
    ref_inp1 = to_reference(inp1, upcast)
    ref_inp2 = to_reference(inp2, upcast)

    ref_out = ref_inp1.div_(ref_inp2, rounding_mode="trunc")
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2, rounding_mode="trunc")

    if not TO_CPU:
        logging.debug(
            f"The maximum difference between torch and triton is "
            f"{torch.max(torch.abs(ref_out - res_out))}"
        )
    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div
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
def test_accuracy_floor_divide_float(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2, rounding_mode="floor")
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode="floor")

    gems_assert_equal(res_out, ref_out, equal_nan=True)


# TODO: failed at large size, eg. (65536 * 2048,)
@pytest.mark.inplace
@pytest.mark.floor_divide_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_floor_divide_float_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.clone(), False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2, rounding_mode="floor")
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2, rounding_mode="floor")

    gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(flag_gems.vendor_name == "aipu", reason="TODO")
@pytest.mark.floor_divide
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_floor_divide_int(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)

    if TO_CPU:
        inp1 = replace_zeros(inp1)
        inp2 = replace_zeros(inp2)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = ref_inp1 // ref_inp2
    with flag_gems.use_gems():
        res_out = inp1 // inp2

    gems_assert_equal(res_out, ref_out)

    for d in inp2.flatten()[:2]:
        ref_d = to_reference(d, False)
        ref_out = ref_inp1 // ref_d
        with flag_gems.use_gems():
            res_out = inp1 // d
        gems_assert_equal(res_out, ref_out)

        ref_out = ref_d // ref_inp1
        with flag_gems.use_gems():
            res_out = d // inp1
        gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.floor_divide_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_floor_divide_int_(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(
        flag_gems.device,
    )
    inp2 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(
        flag_gems.device,
    )
    if TO_CPU:
        inp1 = replace_zeros(inp1)
        inp2 = replace_zeros(inp2)
    ref_inp1 = to_reference(inp1.clone(), False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = ref_inp1.floor_divide_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.floor_divide_(inp2)

    gems_assert_equal(res_out, ref_out)

    ref_inp1 = to_reference(inp1.clone(), False)
    for d in inp2.flatten()[:2]:
        ref_d = to_reference(d, False)
        ref_out = ref_inp1.floor_divide_(ref_d)
        with flag_gems.use_gems():
            res_out = inp1.floor_divide_(d)
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
        device="cpu",
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)
    if TO_CPU:
        inp1 = replace_zeros(inp1)
        inp2 = replace_zeros(inp2)
    ref_inp1 = to_reference(inp1, False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = ref_inp1 % ref_inp2
    with flag_gems.use_gems():
        res_out = inp1 % inp2

    gems_assert_equal(res_out, ref_out)

    for d in inp2.flatten()[:2]:
        ref_d = to_reference(d, False)
        ref_out = ref_inp1 % ref_d
        with flag_gems.use_gems():
            res_out = inp1 % d
        gems_assert_equal(res_out, ref_out)

        ref_out = ref_d % ref_inp1
        with flag_gems.use_gems():
            res_out = d % inp1
        gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.remainder_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_accuracy_remainder_(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    if TO_CPU:
        inp1 = replace_zeros(inp1.clone())
        inp2 = replace_zeros(inp2)
    ref_inp1 = to_reference(inp1.clone(), False)
    ref_inp2 = to_reference(inp2, False)

    ref_out = ref_inp1.remainder_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.remainder_(inp2)

    gems_assert_equal(res_out, ref_out)

    ref_inp1 = to_reference(inp1.clone(), False)
    for d in inp2.flatten()[:2]:
        ref_d = to_reference(d, False)
        ref_out = ref_inp1.remainder_(ref_d)
        with flag_gems.use_gems():
            res_out = inp1.remainder_(d)
        gems_assert_equal(res_out, ref_out)


@pytest.mark.eq
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_eq(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(
        torch.nn.functional.gelu(ref_inp1, approximate=approximate), ref_inp2
    )
    with flag_gems.use_gems():
        res_out = flag_gems.gelu_and_mul(inp1, inp2, approximate)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    (ref_inp1_grad, ref_inp2_grad) = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )

    (res_inp1_grad, res_inp2_grad) = torch.autograd.grad(
        res_out, (inp1, inp2), out_grad
    )

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_inp1_grad, ref_inp1_grad, dtype)
    gems_assert_close(res_inp2_grad, ref_inp2_grad, dtype)


@pytest.mark.gt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_gt(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.mul(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.mul_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.clone(), True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = ref_inp1.mul_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.mul_(inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.mul_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul_tensor_scalar_(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = to_reference(inp1.clone(), True)

    ref_out = ref_inp1.mul_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.mul_(inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_mul_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin" or flag_gems.vendor_name == "ascend":
        inp1 = inp1.uniform_(-1, 1)
        inp2 = inp2.uniform_(-1, 1)

    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.pow(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.pow_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin":
        inp1 = inp1.uniform_(-1, 1)
        inp2 = inp2.uniform_(-1, 1)

    ref_inp1 = to_reference(inp1.clone(), True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = ref_inp1.pow_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.pow_(inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.atan2
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_atan2(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin" or flag_gems.vendor_name == "ascend":
        inp1 = inp1.uniform_(-1, 1)
        inp2 = inp2.uniform_(-1, 1)

    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.atan2(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.atan2(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.maximum
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_maximum(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin" or flag_gems.vendor_name == "ascend":
        inp2 = inp2.uniform_(-1, 1)

    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.pow(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.pow
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "scalar",
    SCALARS + ([1, 2, 3, 4, 5, 8] if flag_gems.vendor_name == "cambricon" else []),
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_tensor_scalar(scalar, shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar

    if flag_gems.vendor_name == "kunlunxin" or flag_gems.vendor_name == "ascend":
        if scalar == -0.999:
            inp1 = inp1.uniform_(-1, 1)
        elif scalar == -111.999 and dtype == torch.float16:
            inp1 = inp1.uniform_(-1, 1)
        else:
            inp1 = inp1.uniform_(-0.1, 0.1)

    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.pow(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.pow(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.pow_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_tensor_scalar_(scalar, shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar

    if flag_gems.vendor_name == "kunlunxin":
        if scalar == -0.999:
            inp1 = inp1.uniform_(-1, 1)
        elif scalar == -111.999 and dtype == torch.float16:
            inp1 = inp1.uniform_(-1, 1)
        else:
            inp1 = inp1.uniform_(-0.1, 0.1)

    ref_inp1 = to_reference(inp1.clone(), True)

    ref_out = ref_inp1.pow_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.pow_(inp2)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.silu_and_mul
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu_and_mul(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mul(torch.nn.functional.silu(ref_inp1), ref_inp2)
    with flag_gems.use_gems():
        res_out = flag_gems.silu_and_mul(inp1, inp2)

    out_grad = torch.randn_like(res_out)
    ref_grad = to_reference(out_grad, True)

    (ref_inp1_grad, ref_inp2_grad) = torch.autograd.grad(
        ref_out, (ref_inp1, ref_inp2), ref_grad
    )

    (res_inp1_grad, res_inp2_grad) = torch.autograd.grad(
        res_out, (inp1, inp2), out_grad
    )

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_inp1_grad, ref_inp1_grad, dtype)
    gems_assert_close(res_inp2_grad, ref_inp2_grad, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = to_reference(inp1, True)

    ref_out = torch.sub(ref_inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.sub_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.clone(), True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = ref_inp1.sub_(ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.sub_(inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.sub_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_tensor_scalar_(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = to_reference(inp1.clone(), True)

    ref_out = ref_inp1.sub_(inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.sub_(inp2, alpha=alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)

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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    cond = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
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
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.where(ref_inp2 > 0, ref_inp2, inp1)
    with flag_gems.use_gems():
        res_out = torch.where(inp2 > 0, inp2, inp1)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.nan_to_num
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("nan", [None, 0.0, 2.3])
@pytest.mark.parametrize("posinf", [None, 999.0])
@pytest.mark.parametrize("neginf", [None, -999.0])
def test_accuracy_nan_to_num(shape, dtype, nan, posinf, neginf):
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    base.view(-1)[0] = float("nan")
    if base.numel() > 1:
        base.view(-1)[1] = float("inf")
    if base.numel() > 2:
        base.view(-1)[2] = float("-inf")

    ref_input = to_reference(base)
    ref_out = torch.nan_to_num(ref_input, nan=nan, posinf=posinf, neginf=neginf)

    with flag_gems.use_gems():
        res_out = torch.nan_to_num(base, nan=nan, posinf=posinf, neginf=neginf)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.isclose
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
# @pytest.mark.parametrize("dtype", FLOAT_DTYPES + ALL_INT_DTYPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
@pytest.mark.parametrize("zero_tol", [False, True])
@pytest.mark.parametrize("equal_nan", [False, True])
@pytest.mark.parametrize("gen_nan", [0, 1, 2, 3, 4])
def test_accuracy_isclose(shape, dtype, zero_tol, equal_nan, gen_nan):
    # [gen_nan] 1: nan, 2: inf, 3: -inf, 4: inf vs -inf
    rtol = (
        torch.rand(1, dtype=torch.float32, device=flag_gems.device).item() * 0.0001
        if not zero_tol
        else 0
    )
    if dtype in ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        if gen_nan:
            nan_num = torch.full(
                (1,),
                float("nan" if gen_nan == 1 else "inf"),
                dtype=dtype,
                device=flag_gems.device,
            )
            inp1.view(-1)[0] = -nan_num if gen_nan == 3 else nan_num
            inp2.view(-1)[0] = -nan_num if gen_nan >= 3 else nan_num
        atol = (
            torch.finfo(dtype).tiny
            * torch.randint(0, 4, (1,), device=flag_gems.device).item()
            if not zero_tol
            else 0
        )
    else:
        inp1 = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
        inp2 = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
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
                    * torch.randint(0, 10, (1,), device=flag_gems.device).item()
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


@pytest.mark.logical_or
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES + BOOL_TYPES)
def test_accuracy_logical_or(shape, dtype):
    if dtype in ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
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
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in ALL_FLOAT_DTYPES:
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
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
        inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in ALL_INT_DTYPES:
        inp1 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.logical_xor(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.logical_xor(inp1, inp2)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.threshold
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_threshold(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)
    threshold = 0
    value = 100

    ref_out = torch.nn.functional.threshold(ref_inp, threshold, value)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.threshold(res_inp, threshold, value)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.threshold
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_threshold_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    threshold = 0

    ref_inp = to_reference(res_inp, True)
    ref_grad = to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.threshold_backward(ref_grad, ref_inp, threshold)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.threshold_backward(res_grad, res_inp, threshold)

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.polar
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_polar(shape, dtype):
    abs = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 5
    angle = (torch.rand(shape, dtype=dtype, device=flag_gems.device) - 0.5) * (
        8 * math.pi
    )
    ref_abs = to_reference(abs)
    ref_angle = to_reference(angle)
    ref_out = torch.polar(ref_abs, ref_angle)
    with flag_gems.use_gems():
        res_out = torch.polar(abs, angle)

    gems_assert_close(res_out.real, ref_out.real, dtype)
    gems_assert_close(res_out.imag, ref_out.imag, dtype)


@pytest.mark.lerp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_lerp(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype is torch.half:
        pytest.skip("wait lerp cpu half impl")

    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)
    weight.uniform_(-0.1, 0.1)

    ref_input = to_reference(input)
    ref_end = to_reference(end)
    ref_weight = to_reference(weight)

    ref_out = torch.lerp(ref_input, ref_end, weight=5.0)
    with flag_gems.use_gems():
        res_out = torch.lerp(input, end, weight=5.0)
    gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.lerp(ref_input, ref_end, weight=ref_weight)
    with flag_gems.use_gems():
        res_out = torch.lerp(input, end, weight=weight)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.lerp_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_lerp_(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype is torch.half:
        pytest.skip("wait lerp cpu half impl")

    torch.manual_seed(0)

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    end = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    input.uniform_(-0.1, 0.1)
    end.uniform_(-0.1, 0.1)
    weight.uniform_(-0.1, 0.1)

    ref_input = to_reference(input)
    ref_end = to_reference(end)
    ref_weight = to_reference(weight)

    ref_out = ref_input.clone().lerp_(ref_end, weight=5.0)
    with flag_gems.use_gems():
        res_out = input.clone().lerp_(end, weight=5.0)
    gems_assert_close(res_out, ref_out, dtype)

    ref_out = ref_input.clone().lerp_(ref_end, weight=ref_weight)
    with flag_gems.use_gems():
        res_out = input.clone().lerp_(end, weight=weight)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize(
    "value",
    [
        torch.tensor(1024, device=flag_gems.device),
        torch.scalar_tensor(1024, device=flag_gems.device),
        1024,
    ],
)
def test_accuracy_masked_fill(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    if torch.is_tensor(value):
        ref_out = torch.masked_fill(ref_inp, ref_mask, to_reference(value))
    else:
        ref_out = torch.masked_fill(ref_inp, ref_mask, value)
    with flag_gems.use_gems():
        res_out = torch.masked_fill(inp, mask, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_fill_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize(
    "value",
    [
        torch.tensor(1024, device=flag_gems.device),
        torch.scalar_tensor(1024, device=flag_gems.device),
        1024,
    ],
)
def test_accuracy_masked_fill_(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.randn(shape, dtype=dtype, device=flag_gems.device) < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    if torch.is_tensor(value):
        ref_inp.masked_fill_(ref_mask, to_reference(value))
    else:
        ref_inp.masked_fill_(ref_mask, value)
    with flag_gems.use_gems():
        inp.masked_fill_(mask, value)

    gems_assert_equal(inp, ref_inp)


@pytest.mark.fill_
@pytest.mark.parametrize("value", [0, 1, 9])
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_fill_(value, shape, dtype):
    # Test fill_.Scalar
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x.clone(), False)

    ref_x.fill_(value)
    with flag_gems.use_gems():
        x.fill_(value)

    gems_assert_equal(x, ref_x)

    # Test fill_.Tensor
    x = torch.ones(shape, device=flag_gems.device, dtype=dtype)
    ref_x = to_reference(x.clone(), False)
    value_tensor = torch.tensor(value, device=flag_gems.device, dtype=dtype)
    if flag_gems.vendor_name == "mthreads":
        ref_x.fill_(value_tensor.cpu())
    else:
        ref_value_tensor = to_reference(value_tensor)
        ref_x.fill_(ref_value_tensor)
    with flag_gems.use_gems():
        x.fill_(value_tensor)

    gems_assert_equal(x, ref_x)


@pytest.mark.addcmul
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_addcmul(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(res_inp, True)
    ref_t1 = to_reference(t1, True)
    ref_t2 = to_reference(t2, True)

    v = float(np.float32(random.random()))

    ref_out = torch.addcmul(ref_inp, ref_t1, ref_t2, value=v)
    with flag_gems.use_gems():
        res_out = torch.addcmul(res_inp, t1, t2, value=v)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.addcdiv
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_addcdiv(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(res_inp, True)
    ref_t1 = to_reference(t1, True)
    ref_t2 = to_reference(t2, True)

    v = float(np.float32(random.random()))

    ref_out = torch.addcdiv(ref_inp, ref_t1, ref_t2, value=v)
    with flag_gems.use_gems():
        res_out = torch.addcdiv(res_inp, t1, t2, value=v)

    gems_assert_close(res_out, ref_out, dtype)
