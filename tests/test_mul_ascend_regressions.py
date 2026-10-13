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

"""Ascend MUL grid traversal and aliased autotuning regressions."""
import importlib

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend kernels"
)


def module():
    return importlib.import_module("flag_gems.runtime.backend._ascend.ops.mul")


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
def test_mul_bounded_grid_complete_traversal(kind):
    m = module()
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


@pytest.mark.parametrize(
    "shape,bshape,dtype",
    [
        ((11, 13), (11, 1), torch.float32),
        ((3, 5, 7), (3, 1, 1), torch.float32),
        ((11, 13), (11, 1), torch.complex64),
    ],
)
def test_mul_inplace_autotuning_first_and_second_call(shape, bshape, dtype):
    m = module()
    av = complex(0.5, 0.25) if dtype.is_complex else 0.5
    bv = complex(0.75, -0.5) if dtype.is_complex else 0.75
    a = torch.full(shape, av, device=flag_gems.device, dtype=dtype)
    b = torch.full(bshape, bv, device=flag_gems.device, dtype=dtype)
    for _ in range(2):
        ref = torch.mul(a, b)
        result = m.mul_(a, b)
        assert result is a
        torch.testing.assert_close(result.cpu(), ref.cpu())


@pytest.mark.parametrize(
    "ad,bd", [(torch.float16, torch.float32), (torch.int32, torch.int64)]
)
def test_mul_inplace_safe_cast(ad, bd):
    m = module()
    a = torch.tensor([2, 3], dtype=ad, device=flag_gems.device)
    b = torch.tensor([3, 4], dtype=bd, device=flag_gems.device)
    ref = a.clone().mul_(b)
    torch.testing.assert_close(m.mul_(a, b).cpu(), ref.cpu())


def test_mul_output_safe_cast():
    m = module()
    a = torch.tensor([2, 3], device=flag_gems.device)
    b = torch.tensor([3, 4], device=flag_gems.device)
    out = torch.empty(2, dtype=torch.float32, device=flag_gems.device)
    ref = torch.mul(a, b, out=torch.empty_like(out))
    assert m.mul(a, b, out=out) is out
    torch.testing.assert_close(out.cpu(), ref.cpu())


def test_mul_inplace_unsafe_cast_rejected():
    m = module()
    a = torch.tensor([2, 3], device=flag_gems.device)
    with pytest.raises(RuntimeError):
        m.mul_(a, 0.75)


@pytest.mark.parametrize(
    "ad,od",
    [
        (torch.float32, torch.complex64),
        (torch.int32, torch.complex64),
        (torch.complex64, torch.complex128),
    ],
)
def test_mul_complex_output_cast(ad, od):
    m = module()
    a = torch.tensor([2, 3], dtype=ad, device=flag_gems.device)
    b = torch.tensor([3, 4], dtype=ad, device=flag_gems.device)
    out = torch.empty(2, dtype=od, device=flag_gems.device)
    ref = torch.mul(a, b, out=torch.empty_like(out))
    assert m.mul(a, b, out=out) is out
    torch.testing.assert_close(out.cpu(), ref.cpu())


@pytest.mark.parametrize(
    "dtype,values", [(torch.int32, [65536, 65537]), (torch.float16, [1.001, 1.003])]
)
def test_mul_complex_out_preserves_real_computation(dtype, values):
    m = module()
    a = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    out = torch.empty(2, dtype=torch.complex64, device=flag_gems.device)
    ref = torch.mul(a, a, out=torch.empty_like(out))
    m.mul(a, a, out=out)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=0, atol=0)


def test_mul_cast_output_shape_error_preserves_output():
    m = module()
    a = torch.tensor([2, 3], dtype=torch.int32, device=flag_gems.device)
    out = torch.full((3,), 7, dtype=torch.complex64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="output with shape"):
        m.mul(a, a, out=out)
    torch.testing.assert_close(out.cpu(), torch.full((3,), 7, dtype=torch.complex64))
