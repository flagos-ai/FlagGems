import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


FORWARD_CASES = [
    pytest.param((1, 1, 1, 1), (1, 1), None, None, id="identity_1x1_output_size"),
    pytest.param((1, 3, 8, 8), (8, 8), None, None, id="identity_output_size"),
    pytest.param((2, 3, 32, 32), (64, 64), None, None, id="integer_x2_output_size"),
    pytest.param((1, 3, 8, 8), (32, 32), None, None, id="integer_x4_output_size"),
    pytest.param(
        (2, 3, 32, 32), (67, 118), None, None, id="non_integer_output_size"
    ),
    pytest.param((1, 3, 8, 8), (10, 40), 1.3, 5.1, id="non_integer_scales"),
    pytest.param((4, 16, 64, 64), (32, 32), None, None, id="downsample_output_size"),
    pytest.param((1, 64, 128, 128), (38, 64), 0.3, 0.5, id="downsample_scales"),
    pytest.param((2, 3, 32, 32), (32, 64), None, None, id="asymmetric_h_same"),
    pytest.param((2, 3, 32, 32), (64, 32), 2.0, 1.0, id="asymmetric_w_same"),
    pytest.param((1, 3, 8, 8), (16, 16), 2.0, 2.0, id="explicit_integer_scales"),
    pytest.param(
        (1, 3, 8, 8), (11, 13), 2.0, 1.5, id="output_size_and_scales"
    ),
    pytest.param(
        (1, 3, 8, 8), (8, 8), 2.0, 2.0, id="same_output_size_non_identity_scales"
    ),
]


def _make_layout_input(shape, dtype, layout):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if layout == "channels_last":
        return input.contiguous(memory_format=torch.channels_last)
    if layout == "transpose":
        return input.transpose(2, 3)
    return input


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shape, output_size, scales_h, scales_w", FORWARD_CASES)
def test_upsample_nearest2d(dtype, shape, output_size, scales_h, scales_w):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input)

    ref_out = torch._C._nn.upsample_nearest2d(
        ref_i, output_size, scales_h, scales_w
    )
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(
            input, output_size, scales_h, scales_w
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["channels_last", "transpose"])
def test_upsample_nearest2d_layout(dtype, layout):
    input = torch.randn((2, 3, 7, 9), dtype=dtype, device=flag_gems.device)
    if layout == "channels_last":
        input = input.contiguous(memory_format=torch.channels_last)
        output_size = (14, 18)
        scales_h = 2.0
        scales_w = 2.0
    else:
        input = input.transpose(2, 3)
        output_size = (13, 11)
        scales_h = None
        scales_w = None

    ref_i = utils.to_reference(input)
    ref_out = torch._C._nn.upsample_nearest2d(
        ref_i, output_size, scales_h, scales_w
    )
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(
            input, output_size, scales_h, scales_w
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out.stride() == ref_out.stride()


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "transpose"])
@pytest.mark.parametrize("scales_h, scales_w", [(None, None), (1.0, 1.0), (2.0, 2.0)])
def test_upsample_nearest2d_identity_fresh_tensor_preserves_layout(
    dtype, layout, scales_h, scales_w
):
    input = _make_layout_input((2, 3, 4, 5), dtype, layout)
    output_size = tuple(input.shape[-2:])
    input_before = input.clone(memory_format=torch.preserve_format)

    ref_out = torch._C._nn.upsample_nearest2d(input, output_size, scales_h, scales_w)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(
            input, output_size, scales_h, scales_w
        )

    assert ref_out.data_ptr() != input.data_ptr()
    assert res_out.data_ptr() != input.data_ptr()
    assert res_out.stride() == ref_out.stride()
    utils.gems_assert_close(res_out, ref_out, dtype)

    res_out.add_(1)
    utils.gems_assert_close(input, input_before, dtype)
