import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


def _nearest2d_reference(ref_i, output_size, scales_h=None, scales_w=None):
    if scales_h is None and scales_w is None:
        return torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size)

    output_h, output_w = output_size
    input_h, input_w = ref_i.shape[2:]
    reciprocal_h = (1.0 / scales_h) if scales_h is not None else (input_h / output_h)
    reciprocal_w = (1.0 / scales_w) if scales_w is not None else (input_w / output_w)
    idx_h = (
        torch.arange(output_h, device=ref_i.device, dtype=torch.float32) * reciprocal_h
    ).to(torch.int64)
    idx_w = (
        torch.arange(output_w, device=ref_i.device, dtype=torch.float32) * reciprocal_w
    ).to(torch.int64)
    idx_h = idx_h.clamp(max=input_h - 1)
    idx_w = idx_w.clamp(max=input_w - 1)
    return ref_i.index_select(2, idx_h).index_select(3, idx_w)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_explicit_scales_x2_output(dtype):
    input = torch.randn((2, 4, 5, 7), dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = (10, 14)

    ref_out = _nearest2d_reference(
        ref_i, output_size=output_size, scales_h=1.7, scales_w=2.3
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(
            input, output_size=output_size, scales_h=1.7, scales_w=2.3
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_nearest2d
def test_upsample_nearest2d_preserves_legacy_nearest_not_nearest_exact():
    input = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32, device=flag_gems.device)
    input = input.reshape(1, 2, 3, 4)
    output_size = (5, 7)

    nearest_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)
    nearest_exact_out = torch.ops.aten._upsample_nearest_exact2d(
        input, output_size, None, None
    )
    assert not torch.equal(nearest_out, nearest_exact_out)

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, nearest_out, torch.float32)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_identity_fresh_copy_and_layout(dtype):
    input = torch.randn((2, 3, 8, 9), dtype=dtype, device=flag_gems.device)
    input = input.contiguous(memory_format=torch.channels_last)
    ref_i = utils.to_reference(input)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=input.shape[2:])

    utils.gems_assert_close(res_out, ref_i, dtype)
    assert res_out.data_ptr() != input.data_ptr()
    assert res_out.is_contiguous(memory_format=torch.channels_last)
    assert not res_out.is_contiguous()


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_c1_ambiguous_channels_last_uses_contiguous(dtype):
    input = torch.randn((2, 1, 5, 7), dtype=dtype, device=flag_gems.device)
    input = input.contiguous(memory_format=torch.channels_last)
    ref_i = utils.to_reference(input)
    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=(10, 14))
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=(10, 14))

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out.stride() == ref_out.stride()
    assert res_out.is_contiguous()


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize(
    ("shape", "output_size", "scales_h", "scales_w"),
    [
        ((2, 4, 5, 7), (11, 9), None, None),
        ((2, 4, 9, 11), (4, 5), None, None),
        ((2, 4, 5, 7), (10, 14), 1.7, 2.3),
        ((1, 17, 7, 5), (13, 11), None, None),
        ((1, 64, 5, 6), (11, 14), 2.1, 2.3),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_channels_last_non_x2_strides(
    dtype, shape, output_size, scales_h, scales_w
):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    input = input.contiguous(memory_format=torch.channels_last)
    ref_i = utils.to_reference(input)
    ref_out = _nearest2d_reference(
        ref_i, output_size=output_size, scales_h=scales_h, scales_w=scales_w
    ).contiguous(memory_format=torch.channels_last)

    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(
            input, output_size=output_size, scales_h=scales_h, scales_w=scales_w
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out.stride() == ref_out.stride()
    assert res_out.is_contiguous(memory_format=torch.channels_last)
    assert not res_out.is_contiguous()


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_spatial_strided_input_contiguous_output(dtype):
    base = torch.randn((2, 4, 5, 7), dtype=dtype, device=flag_gems.device)
    input = base.transpose(2, 3)
    ref_i = utils.to_reference(input).to(torch.float32)

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=(11, 9)).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=(11, 9))

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out.is_contiguous()
