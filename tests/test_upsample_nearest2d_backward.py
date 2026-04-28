import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


BACKWARD_CASES = [
    pytest.param((1, 1, 1, 1), (1, 1), None, None, id="identity_1x1"),
    pytest.param((1, 3, 8, 8), (8, 8), None, None, id="identity"),
    pytest.param((2, 3, 16, 16), (32, 32), None, None, id="integer_x2"),
    pytest.param((1, 3, 8, 8), (32, 32), None, None, id="integer_x4"),
    pytest.param((2, 3, 16, 16), (33, 59), None, None, id="non_integer_output_size"),
    pytest.param((1, 3, 8, 8), (10, 40), 1.3, 5.1, id="non_integer_scales"),
    pytest.param((4, 16, 32, 32), (16, 16), None, None, id="downsample_output_size"),
    pytest.param((1, 64, 64, 64), (19, 32), 0.3, 0.5, id="downsample_scales"),
    pytest.param((2, 3, 16, 16), (16, 32), None, None, id="asymmetric_h_same"),
    pytest.param((2, 3, 16, 16), (32, 16), 2.0, 1.0, id="asymmetric_w_same"),
    pytest.param((1, 3, 8, 8), (11, 13), 2.0, 1.5, id="output_size_and_scales"),
    pytest.param(
        (1, 3, 8, 8), (8, 8), 2.0, 2.0, id="same_output_size_non_identity_scales"
    ),
]


def _make_layout_grad_output(shape, dtype, layout):
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if layout == "channels_last":
        return grad_output.contiguous(memory_format=torch.channels_last)
    if layout == "transpose":
        return grad_output.transpose(2, 3)
    return grad_output


def _contributors_bound(input_size, output_size):
    ih, iw = input_size[-2:]
    oh, ow = output_size
    return max(1, math.ceil(oh / ih) * math.ceil(ow / iw))


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("input_size, output_size, scales_h, scales_w", BACKWARD_CASES)
def test_upsample_nearest2d_backward(
    dtype, input_size, output_size, scales_h, scales_w
):
    grad_output = torch.randn(
        (*input_size[:2], *output_size), dtype=dtype, device=flag_gems.device
    )
    ref_grad_output = utils.to_reference(grad_output)

    ref_grad_input = torch.ops.aten.upsample_nearest2d_backward(
        ref_grad_output, output_size, input_size, scales_h, scales_w
    )
    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.upsample_nearest2d_backward(
            grad_output, output_size, input_size, scales_h, scales_w
        )

    utils.gems_assert_close(
        res_grad_input,
        ref_grad_input,
        dtype,
        reduce_dim=_contributors_bound(input_size, output_size),
    )


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["channels_last", "transpose"])
def test_upsample_nearest2d_backward_layout(dtype, layout):
    input_size = (2, 3, 7, 9)
    output_size = (14, 18)
    grad_output = torch.randn(
        (*input_size[:2], *output_size), dtype=dtype, device=flag_gems.device
    )
    if layout == "channels_last":
        grad_output = grad_output.contiguous(memory_format=torch.channels_last)
    else:
        grad_output = grad_output.transpose(2, 3)
        output_size = tuple(grad_output.shape[-2:])

    ref_grad_output = utils.to_reference(grad_output)
    ref_grad_input = torch.ops.aten.upsample_nearest2d_backward(
        ref_grad_output, output_size, input_size, None, None
    )
    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.upsample_nearest2d_backward(
            grad_output, output_size, input_size, None, None
        )

    utils.gems_assert_close(
        res_grad_input,
        ref_grad_input,
        dtype,
        reduce_dim=_contributors_bound(input_size, output_size),
    )
    assert res_grad_input.stride() == ref_grad_input.stride()


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "transpose"])
@pytest.mark.parametrize("scales_h, scales_w", [(None, None), (1.0, 1.0), (2.0, 2.0)])
def test_upsample_nearest2d_backward_identity_fresh_tensor_preserves_layout(
    dtype, layout, scales_h, scales_w
):
    grad_output = _make_layout_grad_output((2, 3, 4, 5), dtype, layout)
    output_size = tuple(grad_output.shape[-2:])
    input_size = tuple(grad_output.shape)
    grad_output_before = grad_output.clone(memory_format=torch.preserve_format)

    ref_grad_input = torch.ops.aten.upsample_nearest2d_backward(
        grad_output, output_size, input_size, scales_h, scales_w
    )
    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.upsample_nearest2d_backward(
            grad_output, output_size, input_size, scales_h, scales_w
        )

    assert ref_grad_input.data_ptr() != grad_output.data_ptr()
    assert res_grad_input.data_ptr() != grad_output.data_ptr()
    assert res_grad_input.stride() == ref_grad_input.stride()
    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)

    res_grad_input.add_(1)
    utils.gems_assert_close(grad_output, grad_output_before, dtype)
