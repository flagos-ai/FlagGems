import pytest
import torch

from flag_gems.ops.linalg_diagonal import linalg_diagonal


def _make_tensor(shape, dtype=torch.float32, device="cuda"):
    if dtype.is_complex:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return real + 1j * imag
    else:
        return torch.randn(shape, dtype=dtype, device=device)


class TestLinalgDiagonal:
    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
    )
    @pytest.mark.parametrize(
        "shape, dim1, dim2",
        [
            ((256, 256), -2, -1),
            ((512, 512), 0, 1),
            ((128, 256, 256), 1, 2),
            ((64, 128, 128, 128), -1, -2),
            ((10, 20, 30, 40), 0, 3),
            ((5, 7, 11, 13), -3, -1),
            ((32, 64, 64, 64), 2, 3),
            ((2, 3, 4, 5, 6), 0, 4),
            ((2, 3, 4, 5, 6), 1, 3),
        ],
    )
    def test_correctness(self, shape, dim1, dim2, dtype):
        A = _make_tensor(shape, dtype)
        result_gems = linalg_diagonal(A, dim1=dim1, dim2=dim2)
        result_torch = torch.diagonal(A, dim1=dim1, dim2=dim2)
        torch.testing.assert_close(result_gems, result_torch, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "shape, dim1, dim2, offset",
        [
            ((5, 5), 0, 1, 2),
            ((5, 5), 0, 1, -2),
            ((7, 7), -2, -1, 3),
            ((7, 7), -2, -1, -4),
            ((10, 10, 10), 0, 2, 1),
            ((10, 10, 10), 0, 2, -2),
            ((4, 8, 16), 0, 2, 3),
            ((4, 8, 16), 1, 2, -1),
        ],
    )
    def test_offset(self, shape, dim1, dim2, offset, dtype):
        A = _make_tensor(shape, dtype)
        result_gems = linalg_diagonal(A, offset=offset, dim1=dim1, dim2=dim2)
        result_torch = torch.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)
        torch.testing.assert_close(result_gems, result_torch, atol=1e-3, rtol=1e-3)

    def test_empty_diag(self):
        A = torch.randn(3, 4, device="cuda")
        result_gems = linalg_diagonal(A, offset=10, dim1=0, dim2=1)
        result_torch = torch.diagonal(A, offset=10, dim1=0, dim2=1)
        assert result_gems.shape == result_torch.shape
        assert result_gems.numel() == 0

    def test_not_contiguous(self):
        A = torch.randn(4, 5, 6, device="cuda").transpose(0, 2)
        dim1, dim2 = 1, 2
        result_gems = linalg_diagonal(A, dim1=dim1, dim2=dim2)
        result_torch = torch.diagonal(A, dim1=dim1, dim2=dim2)
        torch.testing.assert_close(result_gems, result_torch, atol=1e-3, rtol=1e-3)

    @pytest.mark.skip(reason="Custom Triton kernel does not support autograd yet")
    def test_gradient(self):
        A = torch.randn(5, 5, device="cuda", requires_grad=True)
        result = linalg_diagonal(A)
        loss = result.sum()
        loss.backward()
        assert A.grad is not None

    def test_2d_single_element(self):
        A = torch.tensor([[42.0]], device="cuda")
        result = linalg_diagonal(A)
        expected = torch.diagonal(A)
        torch.testing.assert_close(result, expected)

    def test_large_tensor(self):
        A = torch.randn(128, 128, 128, device="cuda")
        result = linalg_diagonal(A, dim1=1, dim2=2)
        expected = torch.diagonal(A, dim1=1, dim2=2)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
