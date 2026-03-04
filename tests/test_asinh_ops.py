# Test case coverage table:
# ┌─────────────────────┬──────────────────────────────────────┬─────────────────────────┐
# │ Test                │ Shapes / Inputs                      │ Dtypes                  │
# ├─────────────────────┼──────────────────────────────────────┼─────────────────────────┤
# │ test_asinh_shapes   │ (1,1),(8,8),(64,64),(256,256),        │ float16,float32,bfloat16│
# │                     │ (1024,1024),(4096,4096)               │                         │
# │ test_asinh_1d       │ 1D tensors of various sizes          │ float16,float32,bfloat16│
# │ test_asinh_2d       │ 2D tensors (64,64),(256,256)         │ float16,float32,bfloat16│
# │ test_asinh_3d       │ 3D tensors (8,8,8),(16,16,16)        │ float16,float32,bfloat16│
# │ test_asinh_4d       │ 4D tensors (4,4,4,4),(8,8,8,8)      │ float16,float32,bfloat16│
# │ test_asinh_edge     │ NaN, Inf, -Inf, 0, neg, large values │ float32                 │
# │ test_asinh_inplace  │ (1,1),(64,64),(1024,1024)            │ float16,float32,bfloat16│
# │ test_asinh_inplace_ │ edge: NaN, Inf, -Inf, 0, neg, large  │ float32                 │
# │   _edge             │                                      │                         │
# └─────────────────────┴──────────────────────────────────────┴─────────────────────────┘

import pytest
import torch

import flag_gems

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

ATOL = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}
RTOL = {
    torch.float16: 1e-4,
    torch.float32: 1e-4,
    torch.bfloat16: 1e-4,
}

POINTWISE_SHAPES_2D = [
    (1, 1),
    (8, 8),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
]


def assert_close(result, reference, dtype):
    """Assert that result and reference are numerically close for the given dtype."""
    torch.testing.assert_close(
        result.to(torch.float32),
        reference.to(torch.float32),
        atol=ATOL[dtype],
        rtol=RTOL[dtype],
        equal_nan=True,
    )


@pytest.mark.parametrize("shape", POINTWISE_SHAPES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_asinh_shapes(shape, dtype):
    """Test asinh across standard 2D shapes."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.asinh(inp.to(torch.float32)).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)
    assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("size", [1, 8, 64, 256, 1024, 4096])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_asinh_1d(size, dtype):
    """Test asinh on 1D tensors."""
    inp = torch.randn(size, dtype=dtype, device=flag_gems.device)
    ref_out = torch.asinh(inp.to(torch.float32)).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)
    assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(64, 64), (256, 256)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_asinh_2d(shape, dtype):
    """Test asinh on 2D tensors."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.asinh(inp.to(torch.float32)).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)
    assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(8, 8, 8), (16, 16, 16)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_asinh_3d(shape, dtype):
    """Test asinh on 3D tensors."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.asinh(inp.to(torch.float32)).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)
    assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(4, 4, 4, 4), (8, 8, 8, 8)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_asinh_4d(shape, dtype):
    """Test asinh on 4D tensors."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_out = torch.asinh(inp.to(torch.float32)).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.asinh(inp)
    assert_close(res_out, ref_out, dtype)


def test_asinh_edge_cases():
    """Test asinh on edge-case values: NaN, Inf, -Inf, zero, negative, large."""
    dtype = torch.float32
    edge_values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 1e6],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_out = torch.asinh(edge_values)
    with flag_gems.use_gems():
        res_out = torch.asinh(edge_values)
    assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(1, 1), (64, 64), (1024, 1024)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_asinh_inplace(shape, dtype):
    """Test asinh_ in-place variant."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = inp.clone()

    ref_out = torch.asinh_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.asinh_(inp)

    assert_close(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr(), "asinh_ must modify tensor in-place"


def test_asinh_inplace_edge_cases():
    """Test asinh_ in-place on edge-case values."""
    dtype = torch.float32
    edge_values = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 1e6],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = edge_values.clone()
    ref_out = torch.asinh_(ref_inp)

    inp = edge_values.clone()
    with flag_gems.use_gems():
        res_out = torch.asinh_(inp)

    assert_close(res_out, ref_out, dtype)
