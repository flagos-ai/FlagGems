import pytest
import torch

import flag_gems
from flag_gems.runtime.backend._iluvatar.ops.conv2d.conv2d_implicit_gemm import _ixdnn_supported

SHAPE_CONV2D = [
    ((2, 16, 32, 32), (32, 16, 3, 3), 1),
]


@pytest.mark.conv2d_padding
@pytest.mark.skipif(flag_gems.vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.skipif(flag_gems.vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV2D)
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_padding(monkeypatch, shape, kernel, stride, padding, groups, dtype, dilation, bias):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    # 先在测试侧也解析padding，对齐wrapper逻辑
    if isinstance(padding, str):
        if padding == "valid":
            pad_h, pad_w = 0, 0
        elif padding == "same":
            _, _, h, w = shape
            sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
            dh, dw = dilation if isinstance(dilation, tuple) else (dilation, dilation)
            kh, kw = kernel[-2], kernel[-1]
            pad_h_total = (h - 1) * sh + (kh - 1) * dh - h + 1
            pad_w_total = (w - 1) * sw + (kw - 1) * dw - w + 1
            pad_h = pad_h_total // 2
            pad_w = pad_w_total // 2
        pad_tuple = (pad_h, pad_w)
    else:
        pad_tuple = padding

    if not _ixdnn_supported(kernel, pad_tuple, (dilation, dilation), bias):
        pytest.skip("IXDNN does not support this conv2d combo on Iluvatar")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    w = torch.randn(kernel, dtype=dtype, device=flag_gems.device, requires_grad=True)
    if bias:
        b_shape = (kernel[0],)
        b = torch.randn(b_shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    else:
        b = None

    gemm_out = flag_gems.conv2d(inp, w, b, stride, padding, dilation, groups)
    ref_inp = inp.detach().cpu()
    ref_w = w.detach().cpu()
    ref_b = b.detach().cpu() if b is not None else None
    ref_out = torch.nn.functional.conv2d(ref_inp, ref_w, ref_b, stride, padding, dilation, groups)

    torch.testing.assert_close(gemm_out.cpu(), ref_out, atol=1e-2, rtol=1e-2)
