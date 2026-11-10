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
from flag_gems.ops._division_aliases import bind_division_aliases

from . import accuracy_utils as utils


# div.Tensor
@pytest.mark.div_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2)
    res_out = flag_gems.div_tensor(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div_.Tensor
@pytest.mark.div__tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div__tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.div_(ref_inp2)
    res_out = flag_gems.div__tensor(inp1, inp2)

    assert res_out is inp1
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


DIVIDE_MODE_CASES = [
    pytest.param(
        "divide_scalar_mode",
        "divide",
        "Scalar_mode",
        marks=pytest.mark.divide_scalar_mode,
    ),
    pytest.param(
        "divide__scalar_mode",
        "divide_",
        "Scalar_mode",
        marks=pytest.mark.divide__scalar_mode,
    ),
    pytest.param(
        "divide_tensor_mode",
        "divide",
        "Tensor_mode",
        marks=pytest.mark.divide_tensor_mode,
    ),
    pytest.param(
        "divide__tensor_mode",
        "divide_",
        "Tensor_mode",
        marks=pytest.mark.divide__tensor_mode,
    ),
]


def _check_divide_mode_alias(entry, packet, overload, rounding_mode, dtype):
    if packet.endswith("_") and rounding_mode is None and not dtype.is_floating_point:
        pytest.skip("Integer in-place true division cannot cast floating output")
    inp = torch.tensor([-17, -8, -1, 0, 1, 8, 17], device=flag_gems.device, dtype=dtype)
    other = torch.full_like(inp, -3) if overload == "Tensor_mode" else -3
    ref_inp = utils.to_reference(inp.clone(), False)
    ref_other = (
        utils.to_reference(other, False) if isinstance(other, torch.Tensor) else other
    )
    aten_op = getattr(getattr(torch.ops.aten, packet), overload)
    ref_out = aten_op(ref_inp, ref_other, rounding_mode=rounding_mode)
    direct_inp = inp.clone()
    direct_out = getattr(flag_gems, entry)(
        direct_inp, other, rounding_mode=rounding_mode
    )
    with flag_gems.use_gems(include=[entry]):
        result = aten_op(inp, other, rounding_mode=rounding_mode)
    if packet.endswith("_"):
        assert direct_out is direct_inp
        assert result is inp
    utils.gems_assert_close(direct_out, ref_out, ref_out.dtype)
    utils.gems_assert_close(result, ref_out, ref_out.dtype)


# Explicit single-marker wrappers keep the AST-based CI check compatible with
# parameter marks without selecting unrelated entries through pytest -m.
@pytest.mark.divide_scalar_mode
@pytest.mark.parametrize("entry,packet,overload", DIVIDE_MODE_CASES[0:1])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_divide_scalar_mode_alias(entry, packet, overload, rounding_mode, dtype):
    _check_divide_mode_alias(entry, packet, overload, rounding_mode, dtype)


@pytest.mark.divide__scalar_mode
@pytest.mark.parametrize("entry,packet,overload", DIVIDE_MODE_CASES[1:2])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_divide__scalar_mode_alias(entry, packet, overload, rounding_mode, dtype):
    _check_divide_mode_alias(entry, packet, overload, rounding_mode, dtype)


@pytest.mark.divide_tensor_mode
@pytest.mark.parametrize("entry,packet,overload", DIVIDE_MODE_CASES[2:3])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_divide_tensor_mode_alias(entry, packet, overload, rounding_mode, dtype):
    _check_divide_mode_alias(entry, packet, overload, rounding_mode, dtype)


@pytest.mark.divide__tensor_mode
@pytest.mark.parametrize("entry,packet,overload", DIVIDE_MODE_CASES[3:4])
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_divide__tensor_mode_alias(entry, packet, overload, rounding_mode, dtype):
    _check_divide_mode_alias(entry, packet, overload, rounding_mode, dtype)


@pytest.mark.parametrize("entry,packet,overload", DIVIDE_MODE_CASES)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor", "invalid"])
def test_divide_mode_backend_forwarding(entry, packet, overload, rounding_mode):
    assert entry == f"{packet}_{overload.lower()}"
    target = "div_mode_" if packet.endswith("_") else "div_mode"
    first, second, sentinel = object(), object(), object()
    calls = []

    def specialized(A, B, rounding_mode=None):
        calls.append((A, B, rounding_mode))
        return sentinel

    # Canonical aliases capture the specialized backend at binding time;
    # they do not dynamically forward through another canonical div entry.
    registry = vars(flag_gems).copy()
    defaults = registry.copy()
    registry[target] = specialized
    bind_division_aliases(registry, defaults)
    assert registry[entry](first, second, rounding_mode=rounding_mode) is sentinel
    assert calls == [(first, second, rounding_mode)]


# div.Tensor_mode with rounding_mode keyword
@pytest.mark.div_tensor_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode,dtype", DIV_MODE_FLOAT_CASES)
def test_div_tensor_mode_float(shape, rounding_mode, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = _make_nonzero_float_tensor(shape, dtype)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, rounding_mode)
    res_out = flag_gems.div_tensor_mode(inp1, inp2, rounding_mode=rounding_mode)

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

    ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, rounding_mode)
    res_out = flag_gems.div_tensor_mode(inp1, inp2, rounding_mode=rounding_mode)

    utils.gems_assert_equal(res_out, ref_out)


# div_.Tensor_mode with rounding_mode keyword
@pytest.mark.div__tensor_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode,dtype", DIV_MODE_FLOAT_CASES)
def test_div__tensor_mode_float(shape, rounding_mode, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = _make_nonzero_float_tensor(shape, dtype)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, rounding_mode)
    res_out = flag_gems.div__tensor_mode(inp1, inp2, rounding_mode=rounding_mode)

    assert res_out is inp1
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div__tensor_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_div__tensor_mode_int(shape, rounding_mode, dtype):
    inp1 = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = _make_nonzero_int_tensor(shape, dtype)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, rounding_mode)
    res_out = flag_gems.div__tensor_mode(inp1, inp2, rounding_mode=rounding_mode)

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

    ref_out = utils.ref_div_rounding(ref_inp, scalar, rounding_mode)
    res_out = flag_gems.div_scalar_mode(inp, scalar, rounding_mode=rounding_mode)

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

    ref_out = utils.ref_div_rounding(ref_inp, scalar, rounding_mode)
    res_out = flag_gems.div_scalar_mode(inp, scalar, rounding_mode=rounding_mode)

    utils.gems_assert_equal(res_out, ref_out)


# div_.Scalar_mode with rounding_mode keyword
@pytest.mark.div__scalar_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode,dtype", DIV_MODE_FLOAT_CASES)
def test_div__scalar_mode_float(shape, rounding_mode, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = -2.5
    ref_inp = utils.to_reference(inp.clone(), False)

    ref_out = utils.ref_div_rounding(ref_inp, scalar, rounding_mode)
    res_out = flag_gems.div__scalar_mode(inp, scalar, rounding_mode=rounding_mode)

    assert res_out is inp
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div__scalar_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_div__scalar_mode_int(shape, rounding_mode, dtype):
    inp = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    scalar = -3
    ref_inp = utils.to_reference(inp.clone(), False)

    ref_out = utils.ref_div_rounding(ref_inp, scalar, rounding_mode)
    res_out = flag_gems.div__scalar_mode(inp, scalar, rounding_mode=rounding_mode)

    assert res_out is inp
    utils.gems_assert_equal(res_out, ref_out)


# div.Scalar
@pytest.mark.div_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_scalar(shape, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float16:
        pytest.skip("Issue #3796: not working")

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = utils.ref_div_rounding(ref_inp1, inp2, None)
    res_out = flag_gems.div_scalar(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div_.Scalar
@pytest.mark.div__scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div__scalar_tensor_input(shape, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float16:
        pytest.skip("Issue #3796: not working")

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), False)

    ref_out = utils.ref_div_rounding(ref_inp1, inp2, None)
    res_out = flag_gems.div__scalar(inp1, inp2)

    assert res_out is inp1
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div__scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div__scalar(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)

    ref_out = utils.ref_div_rounding(ref_inp, scalar, None)
    res_out = flag_gems.div__scalar(inp, scalar)

    assert res_out is inp
    utils.gems_assert_close(inp, ref_out, dtype, equal_nan=True)


# div.Tensor with a zero-dimensional numerator
@pytest.mark.div_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_scalar_tensor(shape, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float16:
        pytest.skip("Issue #3796: not working")

    inp1 = torch.tensor(scalar, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, None)
    res_out = flag_gems.div_tensor(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# div.Tensor with zero-dimensional operands
@pytest.mark.div_tensor
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_div_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random() + 0.01))
        inp2 = float(np.float32(random.random() + 0.01))
    else:
        inp1 = random.randint(1, 100)
        inp2 = random.randint(1, 100)

    inp1 = torch.tensor(inp1, dtype=dtype, device=flag_gems.device)
    inp2 = torch.tensor(inp2, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.div(ref_inp1, ref_inp2)
    res_out = flag_gems.div_tensor(inp1, inp2)

    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_close(res_out, ref_out, ref_out.dtype)


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
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")

    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, None)
    res_out = flag_gems.div_tensor(inp1, inp2)

    # mthreads does not support torch.isclose for complex types on device,
    # so move to CPU before comparison.
    if flag_gems.vendor_name in ("mthreads", "kunlunxin"):
        res_out = res_out.to("cpu")
        ref_out = ref_out.to("cpu")
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
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")

    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    if complex_dtype == torch.complex64:
        float_dtype = torch.float32
    elif complex_dtype == torch.complex32:
        float_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported complex_dtype: {complex_dtype}")

    inp2 = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)

    # mthreads native torch.div returns incorrect results for complex / float,
    # so compute ref on CPU to get correct baseline.
    # Also, mthreads does not support torch.isclose for complex types on device.
    if flag_gems.vendor_name == "mthreads":
        ref_out = torch.div(inp1.to("cpu"), inp2.to("cpu")).to(dtype=complex_dtype)
    else:
        ref_inp1 = utils.to_reference(inp1, True)
        ref_inp2 = utils.to_reference(inp2, True)
        ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, None)

    res_out = flag_gems.div_tensor(inp1, inp2)

    if flag_gems.vendor_name in ("mthreads", "kunlunxin"):
        res_out = res_out.to("cpu")
        ref_out = ref_out.to("cpu")
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
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randint(1, 20, shape, device=flag_gems.device)

    # mthreads native torch.div returns incorrect results for complex / int,
    # so compute ref on CPU to get correct baseline.
    # Also, mthreads does not support torch.isclose for complex types on device.
    if flag_gems.vendor_name == "mthreads":
        ref_out = torch.div(inp1.to("cpu"), inp2.to("cpu")).to(dtype=complex_dtype)
    else:
        ref_inp1 = utils.to_reference(inp1, True)
        ref_inp2 = utils.to_reference(inp2, True)
        ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, None)

    res_out = flag_gems.div_tensor(inp1, inp2)

    if flag_gems.vendor_name in ("mthreads", "kunlunxin"):
        res_out = res_out.to("cpu")
        ref_out = ref_out.to("cpu")
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
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = 3

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = inp2

    ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, None)
    res_out = flag_gems.div_scalar(inp1, inp2)

    # mthreads does not support torch.isclose for complex types on device,
    # so move to CPU before comparison.
    if flag_gems.vendor_name in ("mthreads", "kunlunxin"):
        res_out = res_out.to("cpu")
        ref_out = ref_out.to("cpu")
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
    res_out = flag_gems.div_out(inp1, inp2, out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_out_tensor_scalar(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar_tensor = torch.tensor(scalar, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    ref_scalar = utils.to_reference(scalar_tensor, False)

    ref_out = torch.empty_like(ref_inp)
    torch.div(ref_inp, ref_scalar, out=ref_out)

    out = torch.empty_like(inp)
    res_out = flag_gems.div_out(inp, scalar_tensor, out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_out_scalar_tensor(shape, scalar, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar_tensor = torch.tensor(scalar, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)
    ref_scalar = utils.to_reference(scalar_tensor, False)

    ref_out = torch.empty_like(ref_inp)
    torch.div(ref_scalar, ref_inp, out=ref_out)

    out = torch.empty_like(inp)
    res_out = flag_gems.div_out(scalar_tensor, inp, out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


# ---------------------------------------------------------------------------
# div_tensor_mode / div__tensor_mode and div_scalar_mode / div__scalar_mode
# Covers aten: div.Tensor_mode, div.Scalar_mode, div_.Tensor_mode,
#              div_.Scalar_mode (and their divide.* aliases)
# ---------------------------------------------------------------------------

ROUNDING_MODES = ["trunc", "floor"]


@pytest.mark.div_tensor_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_tensor_mode(shape, rounding_mode, dtype):
    # div.Tensor_mode: div_tensor_mode(Tensor, Tensor, rounding_mode=...)
    if rounding_mode == "trunc" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(
            "trunc_divide uses libdevice.div_rn which only supports float32/float64"
        )
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if flag_gems.vendor_name == "cambricon":
        inp2 = _make_nonzero_float_tensor(shape, dtype)
    else:
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        # avoid divide-by-zero for floor/trunc modes
        inp2 = torch.where(inp2 >= 0, inp2 + 1e-3, inp2 - 1e-3)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, rounding_mode)

    res_out = flag_gems.div_tensor_mode(inp1, inp2, rounding_mode=rounding_mode)

    # mthreads: floor/trunc division with float16/bfloat16 produces ±1~5 integer
    # boundary errors due to float16 not representing small divisors exactly (e.g.
    # 0.001 becomes 0.0010004 in fp16) and different rounding paths between the
    # Triton kernel and CPU/mthreads native implementations. Use atol=5 for these cases.
    if (
        flag_gems.vendor_name == "mthreads"
        and rounding_mode in ("floor", "trunc")
        and dtype in (torch.float16, torch.bfloat16)
    ):
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=5)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div_scalar_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div_scalar_mode(shape, scalar, rounding_mode, dtype):
    # div.Scalar_mode: div_scalar_mode(Tensor, scalar, rounding_mode=...)
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
        scalar_device = ref_inp.device
        scalar_tensor = torch.tensor(scalar, dtype=dtype, device=scalar_device)
        ref_out = utils.ref_div_rounding(ref_inp, scalar_tensor, rounding_mode)
    else:
        ref_out = utils.ref_div_rounding(ref_inp, scalar, rounding_mode)

    res_out = flag_gems.div_scalar_mode(inp, scalar, rounding_mode=rounding_mode)

    # mthreads: floor/trunc division with float16/bfloat16 produces ±1~5 integer
    # boundary errors due to float16 not representing small divisors exactly (e.g.
    # 0.001 becomes 0.0010004 in fp16) and different rounding paths between the
    # Triton kernel and CPU/mthreads native implementations. Use atol=5 for these cases.
    if (
        flag_gems.vendor_name == "mthreads"
        and rounding_mode in ("floor", "trunc")
        and dtype in (torch.float16, torch.bfloat16)
    ):
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=5)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div__tensor_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div__tensor_mode(shape, rounding_mode, dtype):
    # div_.Tensor_mode: div__tensor_mode(Tensor, Tensor, rounding_mode=...)
    if rounding_mode == "trunc" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(
            "trunc_divide uses libdevice.div_rn which only supports float32/float64"
        )

    # mthreads: floor/trunc division with float16/bfloat16 has precision boundary issues.
    # The mthreads specialization and CPU reference diverge on edge cases with small divisors.
    if (
        flag_gems.vendor_name == "mthreads"
        and rounding_mode in ("floor", "trunc")
        and dtype in (torch.float16, torch.bfloat16)
    ):
        pytest.skip(
            "mthreads: floor/trunc division with float16/bfloat16 has precision issues"
        )
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if flag_gems.vendor_name == "cambricon":
        inp2 = _make_nonzero_float_tensor(shape, dtype)
    else:
        inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.where(inp2 >= 0, inp2 + 1e-3, inp2 - 1e-3)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = utils.ref_div_rounding(ref_inp1, ref_inp2, rounding_mode)

    res_out = flag_gems.div__tensor_mode(inp1, inp2, rounding_mode=rounding_mode)

    # mthreads: floor/trunc division with float16/bfloat16 produces ±1~5 integer
    # boundary errors due to float16 not representing small divisors exactly (e.g.
    # 0.001 becomes 0.0010004 in fp16) and different rounding paths between the
    # Triton kernel and CPU/mthreads native implementations. Use atol=5 for these cases.
    if (
        flag_gems.vendor_name == "mthreads"
        and rounding_mode in ("floor", "trunc")
        and dtype in (torch.float16, torch.bfloat16)
    ):
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=5)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.div__scalar_mode
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("rounding_mode", ROUNDING_MODES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_div__scalar_mode(shape, scalar, rounding_mode, dtype):
    # div_.Scalar_mode: div__scalar_mode(Tensor, scalar, rounding_mode=...)
    if rounding_mode == "trunc" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(
            "trunc_divide uses libdevice.div_rn which only supports float32/float64"
        )
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), False)
    # Same workaround as test_div_scalar_mode: use Tensor_mode reference for
    # float scalars in trunc mode to avoid aten CUDA's approximate-division
    # inaccuracy on the Scalar_mode path.
    if rounding_mode == "trunc" and isinstance(scalar, float):
        scalar_device = ref_inp.device
        scalar_tensor = torch.tensor(scalar, dtype=dtype, device=scalar_device)
        ref_out = utils.ref_div_rounding(ref_inp, scalar_tensor, rounding_mode)
    else:
        ref_out = utils.ref_div_rounding(ref_inp, scalar, rounding_mode)
    res_out = flag_gems.div__scalar_mode(inp, scalar, rounding_mode=rounding_mode)

    # mthreads: floor/trunc division with float16/bfloat16 produces ±1~5 integer
    # boundary errors due to float16 not representing small divisors exactly (e.g.
    # 0.001 becomes 0.0010004 in fp16) and different rounding paths between the
    # Triton kernel and CPU/mthreads native implementations. Use atol=5 for these cases.
    if (
        flag_gems.vendor_name == "mthreads"
        and rounding_mode in ("floor", "trunc")
        and dtype in (torch.float16, torch.bfloat16)
    ):
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True, atol=5)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
