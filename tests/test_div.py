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

import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


# div.Tensor with true_divide
@pytest.mark.div_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div_.Tensor with true_divide_
@pytest.mark.div_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_tensor_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


def _make_nonzero_float_tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.where(inp >= 0, inp + 0.1, inp - 0.1)


def _make_nonzero_int_tensor(shape, dtype):
    inp = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    return torch.where(inp == 0, 1, inp)


DIV_MODE_FLOAT_CASES = (
    [(None, dtype) for dtype in utils.FLOAT_DTYPES]
    + [("floor", dtype) for dtype in utils.FLOAT_DTYPES]
    + [("trunc", torch.float32)]
)


# div.Tensor_mode with rounding_mode keyword
@pytest.mark.div_tensor_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode,dtype", DIV_MODE_FLOAT_CASES)
def test_div_tensor_mode_float(shape, rounding_mode, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = _make_nonzero_float_tensor(shape, dtype)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode=rounding_mode)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_tensor_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_div_tensor_mode_int(shape, rounding_mode, dtype):
    inp1 = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = _make_nonzero_int_tensor(shape, dtype)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, rounding_mode=rounding_mode)

    utils.gems_assert_equal(res_out, ref_out)


# div_.Tensor_mode with rounding_mode keyword
@pytest.mark.div_tensor_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode,dtype", DIV_MODE_FLOAT_CASES)
def test_div_tensor_mode_float_(shape, rounding_mode, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = _make_nonzero_float_tensor(shape, dtype)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2, rounding_mode=rounding_mode)

    assert res_out is inp1
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_tensor_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_div_tensor_mode_int_(shape, rounding_mode, dtype):
    inp1 = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = _make_nonzero_int_tensor(shape, dtype)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2, rounding_mode=rounding_mode)

    assert res_out is inp1
    utils.gems_assert_equal(res_out, ref_out)


# div.Scalar_mode with rounding_mode keyword
@pytest.mark.div_scalar_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode,dtype", DIV_MODE_FLOAT_CASES)
def test_div_scalar_mode_float(shape, rounding_mode, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = -2.5
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.div(ref_inp, scalar, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = torch.div(inp, scalar, rounding_mode=rounding_mode)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_div_scalar_mode_int(shape, rounding_mode, dtype):
    inp = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    scalar = -3
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.div(ref_inp, scalar, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = torch.div(inp, scalar, rounding_mode=rounding_mode)

    utils.gems_assert_equal(res_out, ref_out)


# div_.Scalar_mode with rounding_mode keyword
@pytest.mark.div_scalar_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode,dtype", DIV_MODE_FLOAT_CASES)
def test_div_scalar_mode_float_(shape, rounding_mode, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = -2.5
    ref_inp = utils.to_reference(inp.clone(), False)

    ref_out = ref_inp.div_(scalar, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = inp.div_(scalar, rounding_mode=rounding_mode)

    assert res_out is inp
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_div_scalar_mode_int_(shape, rounding_mode, dtype):
    inp = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    scalar = -3
    ref_inp = utils.to_reference(inp.clone(), False)

    ref_out = ref_inp.div_(scalar, rounding_mode=rounding_mode)
    with flag_gems.use_gems():
        res_out = inp.div_(scalar, rounding_mode=rounding_mode)

    assert res_out is inp
    utils.gems_assert_equal(res_out, ref_out)


# div.Tensor with true_divide
@pytest.mark.div_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_scalar(shape, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float16:
        pytest.skip("Issue #3796: not working")

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.div(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div_.Tensor with true_divide_
@pytest.mark.div_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_scalar_(shape, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float16:
        pytest.skip("Issue #3796: not working")

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), False)

    ref_out = ref_inp1.div_(inp2)
    with flag_gems.use_gems():
        res_out = inp1.div_(inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_scalar_(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)

    ref_out = ref_inp.div_(scalar)
    with flag_gems.use_gems():
        res_out = inp.div_(scalar)

    assert res_out is inp
    utils.gems_assert_close(inp, ref_out, dtype, equal_nan=True)


# div.Scalar with true_divide
@pytest.mark.div_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_scalar_tensor(shape, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float16:
        pytest.skip("Issue #3796: not working")

    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div.Scalar with true_divide
@pytest.mark.div_scalar
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_div_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random() + 0.01))
        inp2 = float(np.float32(random.random() + 0.01))
    else:
        inp1 = random.randint(1, 100)
        inp2 = random.randint(1, 100)

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


# div.Tensor
# Complex
@pytest.mark.div_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_div_complex_complex(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype, equal_nan=True)


# div.Tensor
# Complex
@pytest.mark.div_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_div_complex_float_tensor(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    if complex_dtype == torch.complex64:
        float_dtype = torch.float32
    elif complex_dtype == torch.complex32:
        float_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported complex_dtype: {complex_dtype}")

    inp2 = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype, equal_nan=True)


# div.Tensor
# Complex
@pytest.mark.div_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_div_tensor_int(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randint(1, 20, shape, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype, equal_nan=True)


@pytest.mark.div_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_div_complex_int_scalar(shape, complex_dtype):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = 3

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = inp2

    ref_out = torch.div(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_out_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.empty_like(ref_inp1)
    torch.div(ref_inp1, ref_inp2, out=ref_out)

    out = torch.empty_like(inp1)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_out_tensor_scalar(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.empty_like(ref_inp)
    torch.div(ref_inp, scalar, out=ref_out)

    out = torch.empty_like(inp)
    with flag_gems.use_gems():
        res_out = torch.div(inp, scalar, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_out_scalar_tensor(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.empty_like(ref_inp)
    torch.div(scalar, ref_inp, out=ref_out)

    out = torch.empty_like(inp)
    with flag_gems.use_gems():
        res_out = torch.div(scalar, inp, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


# ---------------------------------------------------------------------------
# div_mode / div_mode_
# Covers aten: div.Tensor_mode, div.Scalar_mode, div_.Tensor_mode,
#              div_.Scalar_mode (and their divide.* aliases)
# ---------------------------------------------------------------------------

ROUNDING_MODES = ["trunc", "floor"]


@pytest.mark.div_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_mode_tensor(shape, rounding_mode, dtype):
    # div.Tensor_mode: div_mode(Tensor, Tensor, rounding_mode=...)
    if rounding_mode == "trunc" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(
            "trunc_divide uses libdevice.div_rn which only supports float32/float64"
        )
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # avoid divide-by-zero for floor/trunc modes
    inp2 = inp2 + torch.sign(inp2).clamp(min=1) * 1e-3
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.ops.aten.div.Tensor_mode(
        ref_inp1, ref_inp2, rounding_mode=rounding_mode
    )
    res_out = flag_gems.ops.div_mode(inp1, inp2, rounding_mode=rounding_mode)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_mode_scalar(shape, scalar, rounding_mode, dtype):
    # div.Scalar_mode: div_mode(Tensor, scalar, rounding_mode=...)
    if rounding_mode == "trunc" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(
            "trunc_divide uses libdevice.div_rn which only supports float32/float64"
        )
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    # For trunc mode, use Tensor_mode reference with the scalar cast to the
    # input dtype. aten's CUDA Scalar_mode path uses approximate division
    # internally, producing off-by-one results near integer boundaries that
    # differ from both CPU and f64 references. Casting the scalar to the same
    # dtype gives the correct IEEE 754 result that our kernel matches.
    if rounding_mode == "trunc" and isinstance(scalar, float):
        scalar_tensor = torch.tensor(scalar, dtype=dtype, device=flag_gems.device)
        ref_out = torch.ops.aten.div.Tensor_mode(
            ref_inp, scalar_tensor, rounding_mode=rounding_mode
        )
    else:
        ref_out = torch.ops.aten.div.Scalar_mode(
            ref_inp, scalar, rounding_mode=rounding_mode
        )
    res_out = flag_gems.ops.div_mode(inp, scalar, rounding_mode=rounding_mode)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_mode_tensor_(shape, rounding_mode, dtype):
    # div_.Tensor_mode: div_mode_(Tensor, Tensor, rounding_mode=...)
    if rounding_mode == "trunc" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(
            "trunc_divide uses libdevice.div_rn which only supports float32/float64"
        )
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = inp2 + torch.sign(inp2).clamp(min=1) * 1e-3
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.ops.aten.div_.Tensor_mode(
        ref_inp1, ref_inp2, rounding_mode=rounding_mode
    )
    res_out = flag_gems.ops.div_mode_(inp1, inp2, rounding_mode=rounding_mode)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_mode_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_mode_scalar_(shape, scalar, rounding_mode, dtype):
    # div_.Scalar_mode: div_mode_(Tensor, scalar, rounding_mode=...)
    if rounding_mode == "trunc" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(
            "trunc_divide uses libdevice.div_rn which only supports float32/float64"
        )
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)
    # Same workaround as test_div_mode_scalar: use Tensor_mode reference for
    # float scalars in trunc mode to avoid aten CUDA's approximate-division
    # inaccuracy on the Scalar_mode path.
    if rounding_mode == "trunc" and isinstance(scalar, float):
        scalar_tensor = torch.tensor(scalar, dtype=dtype, device=flag_gems.device)
        ref_out = torch.ops.aten.div.Tensor_mode(
            ref_inp, scalar_tensor, rounding_mode=rounding_mode
        )
    else:
        ref_out = torch.ops.aten.div_.Scalar_mode(
            ref_inp, scalar, rounding_mode=rounding_mode
        )
    res_out = flag_gems.ops.div_mode_(inp, scalar, rounding_mode=rounding_mode)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
