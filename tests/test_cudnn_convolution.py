import pytest
import torch
import torch.nn.functional as F

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

SHAPE_CUDNN_CONV2D = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (1, 3, 3, 3), 1),
    ((32, 8, 8, 8), (32, 8, 2, 2), 1),
]


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CUDNN_CONV2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
def test_cudnn_convolution_2d(
    shape, kernel, stride, padding, groups, dtype, dilation, monkeypatch
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight)

    ref_out = F.conv2d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=[stride, stride],
        padding=[padding, padding],
        dilation=[dilation, dilation],
        groups=groups,
    )

    with flag_gems.use_gems():
        res_out = torch.cudnn_convolution(
            inp,
            weight,
            padding=[padding, padding],
            stride=[stride, stride],
            dilation=[dilation, dilation],
            groups=groups,
            benchmark=False,
            deterministic=False,
            allow_tf32=False,
        )

    gems_assert_close(res_out, ref_out, dtype)


SHAPE_CUDNN_CONV1D = [
    ((32, 2, 4), (17, 2, 2)),
    ((32, 15, 6), (17, 15, 2)),
    ((64, 64, 64), (128, 64, 7)),
]


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel", SHAPE_CUDNN_CONV1D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cudnn_convolution_1d(shape, kernel, stride, padding, dtype, monkeypatch):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight)

    ref_out = F.conv1d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=[stride],
        padding=[padding],
        dilation=[1],
        groups=1,
    )

    with flag_gems.use_gems():
        res_out = torch.cudnn_convolution(
            inp,
            weight,
            padding=[padding],
            stride=[stride],
            dilation=[1],
            groups=1,
            benchmark=False,
            deterministic=False,
            allow_tf32=False,
        )

    gems_assert_close(res_out, ref_out, dtype)


SHAPE_CUDNN_CONV3D = [
    ((1, 2, 5, 5, 5), (1, 2, 3, 3, 3), 1),
    ((2, 3, 9, 9, 9), (1, 3, 3, 3, 3), 1),
]


@pytest.mark.cudnn_convolution
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CUDNN_CONV3D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
def test_cudnn_convolution_3d(
    shape, kernel, stride, padding, groups, dtype, dilation, monkeypatch
):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    torch.backends.cudnn.allow_tf32 = False
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_weight = to_reference(weight)

    ref_out = F.conv3d(
        ref_inp,
        ref_weight,
        bias=None,
        stride=[stride, stride, stride],
        padding=[padding, padding, padding],
        dilation=[dilation, dilation, dilation],
        groups=groups,
    )

    with flag_gems.use_gems():
        res_out = torch.cudnn_convolution(
            inp,
            weight,
            padding=[padding, padding, padding],
            stride=[stride, stride, stride],
            dilation=[dilation, dilation, dilation],
            groups=groups,
            benchmark=False,
            deterministic=False,
            allow_tf32=False,
        )

    gems_assert_close(res_out, ref_out, dtype)
