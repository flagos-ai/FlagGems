import pytest
import torch

import flag_gems

DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]


def make_input(shape, dtype):
    if dtype.is_floating_point:
        return torch.randn(shape, device=flag_gems.device, dtype=dtype)
    return torch.randint(-512, 512, shape, device=flag_gems.device).to(dtype)


def assert_cast(x, non_blocking=False):
    expected = torch.ops.aten._cast_Float(x, non_blocking)
    actual = flag_gems.cast_Float(x, non_blocking)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert actual.stride() == expected.stride()
    assert actual.requires_grad == expected.requires_grad
    if x.dtype == torch.float32:
        assert actual is x
    elif x.numel():
        assert actual.data_ptr() != x.data_ptr()
    return actual, expected


@pytest.mark.cast_Float
@pytest.mark.parametrize(
    "shape", [(), (0,), (2, 0, 3), (1,), (1023,), (1025,), (129, 257)]
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cast_Float(shape, dtype, non_blocking):
    assert_cast(make_input(shape, dtype), non_blocking)


@pytest.mark.cast_Float
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "layout", ["transpose", "slice", "expand", "channels_last", "permuted_slice"]
)
def test_cast_Float_strided(dtype, layout):
    x = make_input((4, 5, 6, 7), dtype)
    if layout == "transpose":
        x = x.transpose(0, 3)
    elif layout == "slice":
        x = x[1:, :, ::2, 1::2]
    elif layout == "expand":
        x = x[:1].expand(9, -1, -1, -1)
    elif layout == "permuted_slice":
        x = x.transpose(0, 3)[::2, :, 1::2, :]
    else:
        x = x.contiguous(memory_format=torch.channels_last)
    assert_cast(x)


@pytest.mark.cast_Float
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_cast_Float_special_values(dtype):
    info = torch.finfo(dtype)
    x = torch.tensor(
        [
            -0.0,
            0.0,
            -1.9,
            0.5,
            1.9,
            info.tiny,
            -info.tiny,
            info.tiny * info.eps,
            info.max,
            -info.max,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        device=flag_gems.device,
        dtype=dtype,
    )
    actual, expected = assert_cast(x)
    assert torch.equal(torch.signbit(actual[:-1]), torch.signbit(expected[:-1]))


@pytest.mark.cast_Float
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cast_Float_integer_limits(dtype):
    info = torch.iinfo(dtype)
    values = [info.min, info.max, -(2**24) - 1, 2**24 + 1]
    if dtype == torch.int64:
        values += [2**53 - 1, 2**53 + 1, -(2**53) - 1]
    assert_cast(torch.tensor(values, device=flag_gems.device, dtype=dtype))


@pytest.mark.cast_Float
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_cast_Float_autograd(dtype, non_blocking):
    x = torch.randn(17, device=flag_gems.device, dtype=dtype, requires_grad=True)
    actual, expected = assert_cast(x, non_blocking)
    grad = torch.randn_like(expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, grad)[0],
        torch.autograd.grad(expected, x, grad)[0],
        rtol=0,
        atol=0,
    )
    with torch.no_grad():
        assert_cast(x, non_blocking)


@pytest.mark.cast_Float
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_cast_Float_complex(dtype):
    assert_cast(torch.randn(17, device=flag_gems.device, dtype=dtype).conj())


@pytest.mark.cast_Float
def test_cast_Float_negative_view():
    assert_cast(torch._neg_view(make_input((17,), torch.float64)))


@pytest.mark.cast_Float
@pytest.mark.parametrize("dtype", DTYPES)
def test_cast_Float_cpu(dtype):
    assert_cast(make_input((17,), dtype).cpu())


@pytest.mark.cast_Float
def test_cast_Float_sparse():
    x = make_input((4, 5), torch.float64).to_sparse()
    expected = torch.ops.aten._cast_Float(x)
    actual = flag_gems.cast_Float(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
