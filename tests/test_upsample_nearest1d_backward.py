import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE, TO_CPU

# QUICK_MODE limits to float32 for fast smoke runs; full runs cover
# utils.FLOAT_DTYPES plus float64 since the native reference supports double.
DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES + [torch.float64]


# Core shapes exercised by SHAPES (mirrors worktree CI branch).
SHAPES = (
    [(1, 2, 7, 13)]
    if QUICK_MODE
    else [
        (1, 1, 1, 9),
        (1, 2, 7, 13),
        (2, 3, 17, 8),
        (4, 8, 64, 127),
    ]
)


def _make_grad_output(shape, dtype, noncontiguous=False):
    batch, channels, _, output_w = shape
    if noncontiguous:
        storage = torch.randn(
            (batch, channels, output_w * 2),
            dtype=dtype,
            device=flag_gems.device,
        )
        return storage[..., ::2]
    return torch.randn(
        (batch, channels, output_w),
        dtype=dtype,
        device=flag_gems.device,
    )


@pytest.mark.upsample_nearest1d_backward
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_nearest1d_backward(shape, dtype, noncontiguous):
    batch, channels, input_w, output_w = shape
    grad_output = _make_grad_output(shape, dtype, noncontiguous)
    ref_grad_output = utils.to_reference(grad_output)
    output_size = (output_w,)
    input_size = (batch, channels, input_w)

    reference = torch.ops.aten.upsample_nearest1d_backward.default(
        ref_grad_output, output_size, input_size
    )
    result = flag_gems.upsample_nearest1d_backward(grad_output, output_size, input_size)

    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.upsample_nearest1d_backward
@pytest.mark.parametrize(
    "input_w,output_w,scale", [(3, 5, 1.6), (4, 7, 2.0), (9, 4, 0.5)]
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_nearest1d_backward_with_scale(input_w, output_w, scale, dtype):
    grad_output = torch.randn((2, 3, output_w), dtype=dtype, device=flag_gems.device)
    ref_grad_output = utils.to_reference(grad_output)
    output_size = (output_w,)
    input_size = (2, 3, input_w)

    reference = torch.ops.aten.upsample_nearest1d_backward.default(
        ref_grad_output, output_size, input_size, scale
    )
    result = flag_gems.upsample_nearest1d_backward(
        grad_output, output_size, input_size, scale
    )

    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.upsample_nearest1d_backward
@pytest.mark.parametrize("shape", [(0, 3, 7, 13), (2, 0, 7, 13)])
def test_upsample_nearest1d_backward_empty(shape):
    batch, channels, input_w, output_w = shape
    grad_output = torch.empty(shape[:2] + (output_w,), device=flag_gems.device)
    ref_grad_output = utils.to_reference(grad_output)
    input_size = (batch, channels, input_w)
    reference = torch.ops.aten.upsample_nearest1d_backward.default(
        ref_grad_output, (output_w,), input_size
    )
    result = flag_gems.upsample_nearest1d_backward(grad_output, (output_w,), input_size)
    utils.gems_assert_equal(result, reference)


@pytest.mark.upsample_nearest1d_backward
@pytest.mark.skipif(TO_CPU, reason="native CPU backward does not support uint8")
def test_upsample_nearest1d_backward_uint8():
    grad_output = torch.randint(
        0, 256, (2, 3, 17), dtype=torch.uint8, device=flag_gems.device
    )
    reference = torch.ops.aten.upsample_nearest1d_backward.default(
        grad_output, (17,), (2, 3, 5)
    )
    result = flag_gems.upsample_nearest1d_backward(grad_output, (17,), (2, 3, 5))
    utils.gems_assert_equal(result, reference)


@pytest.mark.upsample_nearest1d_backward
@pytest.mark.parametrize(
    "output_size,input_size",
    [((0,), (1, 1, 3)), ((6,), (1, 1, 0)), ((6, 7), (1, 1, 3))],
)
def test_upsample_nearest1d_backward_invalid_size(output_size, input_size):
    grad_output = torch.randn((1, 1, max(output_size[0], 1)), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.upsample_nearest1d_backward(grad_output, output_size, input_size)
