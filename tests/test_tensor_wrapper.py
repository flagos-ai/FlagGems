import torch
import triton
from triton import language as tl

from flag_gems.utils import tensor_wrapper


@triton.jit
def double(in_ptr, out_ptr, n, TILE_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = offsets < n
    x = tl.load(in_ptr + offsets, mask=mask)
    out = x * 2.0
    tl.store(out_ptr + offsets, out, mask=mask)


def test_typed_pointer():
    real = torch.randn(10, 10, device="cuda")
    imag = torch.randn(10, 10, device="cuda")
    x = torch.complex(real, imag)

    out = torch.empty_like(x)
    TILE_SIZE = 128
    n = x.numel() * 2
    grid = (
        triton.cdiv(n, TILE_SIZE),
        1,
    )
    in_ptr = tensor_wrapper.TypedPtr(x.data_ptr(), dtype=x.dtype.to_real())
    out_ptr = tensor_wrapper.TypedPtr(out.data_ptr(), dtype=out.dtype.to_real())
    double[grid](in_ptr, out_ptr, n, TILE_SIZE)
    torch.testing.assert_close(out, x * 2.0)


def test_typed_pointer_reinterpret_with_offset():
    real = torch.randn(100, device="cuda")
    imag = torch.randn(100, device="cuda")
    x = torch.complex(real, imag)

    out = torch.empty_like(x)
    TILE_SIZE = 128
    k = 10
    n = (x.numel() - k) * 2
    grid = (
        triton.cdiv(n, TILE_SIZE),
        1,
    )
    in_ptr = tensor_wrapper.TypedPtr.reinterpret_tensor(x, x.dtype.to_real(), 2 * k)
    out_ptr = tensor_wrapper.TypedPtr.reinterpret_tensor(
        out, out.dtype.to_real(), 2 * k
    )
    double[grid](in_ptr, out_ptr, n, TILE_SIZE)
    torch.testing.assert_close(out[k:], x[k:] * 2.0)


def test_typed_pointer_as_is():
    x = torch.randn(100, device="cuda")
    out = torch.empty_like(x)
    TILE_SIZE = 128
    k = 10
    n = x.numel() - k
    grid = (
        triton.cdiv(n, TILE_SIZE),
        1,
    )
    in_ptr = tensor_wrapper.TypedPtr.from_tensor(x, k)
    out_ptr = tensor_wrapper.TypedPtr.from_tensor(out, k)
    double[grid](in_ptr, out_ptr, n, TILE_SIZE)
    torch.testing.assert_close(out[k:], x[k:] * 2.0)


def test_strided_buffer_slice():
    x = torch.randn(100, 100, device="cuda")
    x_buffer = tensor_wrapper.StridedBuffer(x, (10, 10), (100, 1))
    assert x_buffer.size() == (10, 10)
    assert x.element_size() == x.element_size()
    assert x.dim() == 2
