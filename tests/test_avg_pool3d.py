import pytest
import torch
import torch.nn.functional as F

from flag_gems.ops.avg_pool3d import avg_pool3d


class TestAvgPool3d:
    @pytest.mark.parametrize("shape", [(2, 3, 16, 32, 32), (1, 4, 8, 16, 16)])
    @pytest.mark.parametrize("kernel_size", [(2, 2, 2), (3, 3, 3)])
    @pytest.mark.parametrize("stride", [None, (1, 1, 1), (2, 2, 2)])
    @pytest.mark.parametrize("padding", [(0, 0, 0), (1, 1, 1)])
    def test_avg_pool3d_correctness(self, shape, kernel_size, stride, padding):
        in_d, in_h, in_w = shape[2], shape[3], shape[4]
        kd, kh, kw = kernel_size
        sd, sh, sw = stride if stride is not None else kernel_size
        pd, ph, pw = padding

        out_d = (in_d + 2 * pd - kd) // sd + 1
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1

        if out_d <= 0 or out_h <= 0 or out_w <= 0:
            return

        input_tensor = torch.randn(shape, dtype=torch.float32, device="cuda")

        if stride is None:
            stride = kernel_size
        expected = F.avg_pool3d(input_tensor, kernel_size, stride, padding)
        actual = avg_pool3d(input_tensor, kernel_size, stride, padding)

        assert torch.allclose(actual, expected, rtol=1e-4, atol=1.3e-6)

    def test_avg_pool3d_empty(self):
        input_tensor = torch.randn(0, 3, 8, 8, 8, device="cuda")
        output = avg_pool3d(input_tensor, (2, 2, 2))
        assert output.numel() == 0
