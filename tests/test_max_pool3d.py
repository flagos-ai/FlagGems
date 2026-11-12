import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Representative 5-D pooling configs: (shape, kernel_size, stride, padding,
# dilation, ceil_mode). Chosen to cover cubic/non-cubic kernels, strides,
# symmetric/asymmetric padding, dilation, ceil_mode and a typical 3D-CNN shape.
MAXPOOL3D_CONFIGS = [
    # Classic 3x3x3 kernel, stride 2, padding 1
    ((4, 3, 16, 16, 16), 3, 2, 1, 1, False),
    # Non-cubic kernel and stride
    ((8, 16, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
    # ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, 1, True),
    # dilation
    ((1, 1, 9, 9, 9), 2, 1, 0, 2, False),
    # Typical 3D CNN shape
    ((1, 64, 8, 28, 28), 3, 2, 1, 1, False),
    # No padding
    ((2, 8, 8, 16, 16), 2, 2, 0, 1, False),
    # Non-symmetric padding
    ((2, 8, 10, 16, 20), 2, 2, (0, 1, 0), 1, False),
    # Small input
    ((1, 1, 5, 5, 5), 2, 1, 0, 1, False),
    # Large batch
    ((8, 16, 8, 8, 8), 3, 1, 1, 1, False),
]


@pytest.mark.max_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode", MAXPOOL3D_CONFIGS
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_pool3d(shape, kernel_size, stride, padding, dilation, ceil_mode, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.max_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    res_out = flag_gems.max_pool3d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


# End-to-end gradient parity for max_pool3d's autograd path, covering both a
# contiguous and a channels_last_3d input. The channels_last_3d case pins the
# backward wrapper's layout handling: the scatter kernel assumes NCDHW
# contiguous storage, so a gradient allocated in NDHWC order would silently
# scatter to wrong addresses.
@pytest.mark.max_pool3d
@pytest.mark.parametrize(
    "memory_format", [torch.contiguous_format, torch.channels_last_3d]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_pool3d_autograd(dtype, memory_format):
    shape = (2, 8, 8, 16, 16)
    kernel_size, stride, padding = 3, 2, 1

    inp = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    ).to(memory_format=memory_format)
    inp.requires_grad_(True)
    ref_inp = utils.to_reference(inp, True).contiguous()

    res_out = flag_gems.max_pool3d(
        inp, kernel_size=kernel_size, stride=stride, padding=padding
    )
    ref_out = torch.max_pool3d(
        ref_inp, kernel_size=kernel_size, stride=stride, padding=padding
    )

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)

    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
