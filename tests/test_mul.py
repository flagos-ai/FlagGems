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

import importlib
import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.mul(ref_inp1, inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_mul_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random()))
        inp2 = float(np.float32(random.random()))
    else:
        inp1 = random.randint(0, 100)
        inp2 = random.randint(0, 100)

    ref_out = torch.mul(inp1, inp2)
    res_out = flag_gems.mul(inp1, inp2)

    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_tensor_tensor_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.mul_(ref_inp2)
    res_out = flag_gems.mul_(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_tensor_scalar_(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), True)

    ref_out = ref_inp1.mul_(inp2)
    res_out = flag_gems.mul_(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((10, 1), (1, 5)),
        ((1, 5), (10, 1)),
        ((1048576, 1), (1, 32)),
        ((3, 1, 5), (1, 4, 1)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_broadcast_shape(shape_a, shape_b, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    assert res_out.shape == ref_out.shape, (
        f"Shape mismatch: FlagGems produced {res_out.shape}, "
        f"expected {ref_out.shape}"
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
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
def test_mul_complex_complex(shape, complex_dtype):
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    # inp1: complex tensor
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    if flag_gems.vendor_name == "cambricon" and complex_dtype == torch.complex32:
        from .accuracy_utils import to_cpu

        res_out = to_cpu(res_out, ref_out)
        ref_out = ref_out.to(complex_dtype)
        torch.testing.assert_close(res_out, ref_out, atol=5e-3, rtol=2e-3)
    else:
        utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.mul
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
def test_mul_complex_float_tensor(shape, complex_dtype):
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    # inp1: complex tensor
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    if complex_dtype == torch.complex64:
        float_dtype = torch.float32
    elif complex_dtype == torch.complex32:
        float_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported complex_dtype: {complex_dtype}")

    inp2 = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)

    # mthreads torch.mul unsupport complex x float.
    if flag_gems.vendor_name == "mthreads":
        ref_inp1 = inp1.to("cpu")
        ref_inp2 = inp2.to("cpu")
    else:
        ref_inp1 = utils.to_reference(inp1, True)
        ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    if flag_gems.vendor_name == "mthreads":
        res_out = res_out.to("cpu")
        ref_out = ref_out.to(complex_dtype)
    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.mul
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
def test_mul_complex_int_tensor(shape, complex_dtype):
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    # inp1: complex tensor
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randint(10, 20, shape, device=flag_gems.device)

    # mthreads torch.mul unsupport complex x int.
    if flag_gems.vendor_name == "mthreads":
        ref_inp1 = inp1.to("cpu")
        ref_inp2 = inp2.to("cpu")
    else:
        ref_inp1 = utils.to_reference(inp1, True)
        ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    if flag_gems.vendor_name == "mthreads":
        res_out = res_out.to("cpu")
        ref_out = ref_out.to(complex_dtype)
    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.mul
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
def test_mul_complex_int_scalar(shape, complex_dtype):
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    # inp1: complex tensor
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = 3

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = inp2

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype)


def _ascend_mul_module():
    return importlib.import_module("flag_gems.runtime.backend._ascend.ops.mul")


@pytest.mark.mul
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend kernels")
@pytest.mark.parametrize(
    "kind",
    [
        "flat",
        "full",
        "scalar",
        "broadcast",
        "generic",
        "runtime",
        "complex",
        "complex_runtime",
        "tiled",
        "chunked",
    ],
)
def test_ascend_mul_bounded_grid_complete_traversal(kind):
    m = _ascend_mul_module()
    n = 1280 if kind == "full" else 1283
    a = torch.linspace(-1, 1, n, device=flag_gems.device)
    b = torch.linspace(0.25, 0.75, n, device=flag_gems.device)
    out = torch.empty_like(a)
    kw = dict(BLOCK_SIZE=256, IS_BOOL=False)
    if kind in ("flat", "full"):
        fn = m.mul_kernel if kind == "flat" else m.mul_contiguous_full_kernel
        fn.jit_function[(2,)](a, b, out, n, dtype="float32", **kw)
        ref = torch.mul(a, b)
    elif kind == "scalar":
        m.mul_scalar_kernel.jit_function[(2,)](a, out, 0.75, n, dtype="float32", **kw)
        ref = torch.mul(a, 0.75)
    elif kind == "broadcast":
        m.mul_broadcast_2d_kernel.jit_function[(2,)](
            a,
            b,
            out,
            n,
            1,
            a_s0=1,
            a_s1=0,
            b_s0=1,
            b_s1=0,
            out_s0=1,
            out_s1=0,
            dtype="float32",
            **kw,
        )
        ref = torch.mul(a, b)
    elif kind in ("generic", "runtime"):
        if kind == "generic":
            m.mul_generic_nd_kernel.jit_function[(2,)](
                a,
                b,
                out,
                n,
                SHAPE=(n,),
                A_STRIDE=(1,),
                B_STRIDE=(1,),
                OUT_STRIDE=(1,),
                NDIM=1,
                dtype="float32",
                **kw,
            )
        else:
            meta = torch.tensor([n, 1, 1, 1], device=flag_gems.device)
            m.mul_generic_nd_runtime_meta_kernel.jit_function[(2,)](
                a, b, out, meta, n, NDIM=1, dtype="float32", **kw
            )
        ref = torch.mul(a, b)
    elif kind in ("complex", "complex_runtime"):
        ac = torch.complex(a, b)
        bc = torch.complex(b, a)
        out = torch.empty_like(ac)
        ar, ai, br, bi = ac.real, ac.imag, bc.real, bc.imag
        if kind == "complex":
            m.mul_complex_generic_nd_kernel.jit_function[(2,)](
                ar,
                ai,
                br,
                bi,
                out.real,
                out.imag,
                n,
                SHAPE=(n,),
                AR_STRIDE=(2,),
                AI_STRIDE=(2,),
                BR_STRIDE=(2,),
                BI_STRIDE=(2,),
                OUT_R_STRIDE=(2,),
                OUT_I_STRIDE=(2,),
                NDIM=1,
                dtype="complex64",
                BLOCK_SIZE=256,
            )
        else:
            meta = torch.tensor([n, 2, 2, 2, 2, 2, 2], device=flag_gems.device)
            m.mul_complex_generic_nd_runtime_meta_kernel.jit_function[(2,)](
                ar,
                ai,
                br,
                bi,
                out.real,
                out.imag,
                meta,
                n,
                NDIM=1,
                dtype="complex64",
                BLOCK_SIZE=256,
            )
        ref = torch.mul(ac, bc)
    elif kind == "tiled":
        a = a[:17].reshape(17, 1)
        b = b[:37].reshape(1, 37)
        out = torch.empty((17, 37), device=flag_gems.device)
        m.mul_broadcast_tiled_kernel.jit_function[(2,)](
            a,
            b,
            out,
            17,
            37,
            a_s0=1,
            a_s1=0,
            b_s0=0,
            b_s1=1,
            dtype="float32",
            BLOCK_M=8,
            BLOCK_N=32,
        )
        ref = torch.mul(a, b)
    else:
        m.mul_contiguous_chunked_kernel.jit_function[(2,)](
            a, b, out, n, BLOCKS_PER_PROGRAM=3, **kw
        )
        ref = torch.mul(a, b)
    torch.testing.assert_close(out.cpu(), ref.cpu())


@pytest.mark.mul
@pytest.mark.mul_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend kernels")
@pytest.mark.parametrize(
    "shape,bshape,dtype",
    [
        ((11, 13), (11, 1), torch.float32),
        ((3, 5, 7), (3, 1, 1), torch.float32),
        ((11, 13), (11, 1), torch.complex64),
    ],
)
def test_ascend_mul_inplace_autotuning_first_and_second_call(shape, bshape, dtype):
    m = _ascend_mul_module()
    av = complex(0.5, 0.25) if dtype.is_complex else 0.5
    bv = complex(0.75, -0.5) if dtype.is_complex else 0.75
    a = torch.full(shape, av, device=flag_gems.device, dtype=dtype)
    b = torch.full(bshape, bv, device=flag_gems.device, dtype=dtype)
    for _ in range(2):
        ref = torch.mul(a, b)
        result = m.mul_(a, b)
        assert result is a
        torch.testing.assert_close(result.cpu(), ref.cpu())


@pytest.mark.mul
@pytest.mark.mul_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend kernels")
@pytest.mark.parametrize(
    "ad,bd", [(torch.float16, torch.float32), (torch.int32, torch.int64)]
)
def test_ascend_mul_inplace_safe_cast(ad, bd):
    m = _ascend_mul_module()
    a = torch.tensor([2, 3], dtype=ad, device=flag_gems.device)
    b = torch.tensor([3, 4], dtype=bd, device=flag_gems.device)
    ref = a.clone().mul_(b)
    torch.testing.assert_close(m.mul_(a, b).cpu(), ref.cpu())


@pytest.mark.mul
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend kernels")
def test_ascend_mul_output_safe_cast():
    m = _ascend_mul_module()
    a = torch.tensor([2, 3], device=flag_gems.device)
    b = torch.tensor([3, 4], device=flag_gems.device)
    out = torch.empty(2, dtype=torch.float32, device=flag_gems.device)
    ref = torch.mul(a, b, out=torch.empty_like(out))
    assert m.mul(a, b, out=out) is out
    torch.testing.assert_close(out.cpu(), ref.cpu())


@pytest.mark.mul
@pytest.mark.mul_
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend kernels")
def test_ascend_mul_inplace_unsafe_cast_rejected():
    m = _ascend_mul_module()
    a = torch.tensor([2, 3], device=flag_gems.device)
    with pytest.raises(RuntimeError):
        m.mul_(a, 0.75)


@pytest.mark.mul
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend kernels")
@pytest.mark.parametrize(
    "ad,od",
    [
        (torch.float32, torch.complex64),
        (torch.int32, torch.complex64),
        (torch.complex64, torch.complex128),
    ],
)
def test_ascend_mul_complex_output_cast(ad, od):
    m = _ascend_mul_module()
    a = torch.tensor([2, 3], dtype=ad, device=flag_gems.device)
    b = torch.tensor([3, 4], dtype=ad, device=flag_gems.device)
    out = torch.empty(2, dtype=od, device=flag_gems.device)
    ref = torch.mul(a, b, out=torch.empty_like(out))
    assert m.mul(a, b, out=out) is out
    torch.testing.assert_close(out.cpu(), ref.cpu())


@pytest.mark.mul
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend kernels")
@pytest.mark.parametrize(
    "dtype,values", [(torch.int32, [65536, 65537]), (torch.float16, [1.001, 1.003])]
)
def test_ascend_mul_complex_out_preserves_real_computation(dtype, values):
    m = _ascend_mul_module()
    a = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    out = torch.empty(2, dtype=torch.complex64, device=flag_gems.device)
    ref = torch.mul(a, a, out=torch.empty_like(out))
    m.mul(a, a, out=out)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=0, atol=0)


@pytest.mark.mul
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend kernels")
def test_ascend_mul_cast_output_shape_error_preserves_output():
    m = _ascend_mul_module()
    a = torch.tensor([2, 3], dtype=torch.int32, device=flag_gems.device)
    out = torch.full((3,), 7, dtype=torch.complex64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="output with shape"):
        m.mul(a, a, out=out)
    torch.testing.assert_close(out.cpu(), torch.full((3,), 7, dtype=torch.complex64))
