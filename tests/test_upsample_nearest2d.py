import os

import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference

SHAPE_SMALL = [
    ((1, 1, 1, 1), (1, 1)),
    ((1, 1, 1, 1), (2, 2)),
    ((1, 1, 8, 8), (8, 8)),
    ((1, 1, 8, 8), (16, 16)),
    ((1, 1, 8, 8), (4, 4)),
    ((1, 2, 1, 1), (2, 2)),
    ((2, 1, 8, 8), (16, 16)),
]

SHAPE_MEDIUM = [
    ((1, 1, 64, 64), (64, 64)),
    ((1, 1, 64, 64), (128, 128)),
    ((1, 1, 64, 64), (32, 32)),
    ((4, 3, 64, 64), (128, 128)),
    ((8, 16, 64, 64), (128, 128)),
    ((1, 1, 256, 256), (256, 256)),
    ((1, 1, 256, 256), (512, 512)),
    ((1, 1, 256, 256), (128, 128)),
    ((4, 3, 256, 256), (512, 512)),
    ((2, 8, 128, 128), (256, 256)),
]

SHAPE_LARGE = [
    ((1, 1, 1024, 1024), (1024, 1024)),
    ((1, 1, 1024, 1024), (2048, 2048)),
    ((1, 1, 1024, 1024), (512, 512)),
    ((2, 3, 1024, 1024), (2048, 2048)),
    ((1, 1, 4096, 4096), (4096, 4096)),
    ((1, 1, 4096, 4096), (2048, 2048)),
]


SHAPE_DIMENSIONS = [
    ((1, 1, 32, 32), (64, 64)),
    ((1, 3, 32, 32), (64, 64)),
    ((1, 16, 32, 32), (64, 64)),
    ((1, 64, 32, 32), (64, 64)),
    ((2, 1, 32, 32), (64, 64)),
    ((4, 1, 32, 32), (64, 64)),
    ((8, 1, 32, 32), (64, 64)),
    ((2, 3, 32, 32), (64, 64)),
    ((4, 8, 32, 32), (64, 64)),
    ((8, 16, 32, 32), (64, 64)),
    ((16, 32, 32, 32), (64, 64)),
]


SHAPE_OUTPUT_SIZE_ONLY = [
    ((1, 1, 32, 32), (32, 32), None, None),
    ((1, 1, 32, 32), (64, 64), None, None),
    ((1, 1, 32, 32), (16, 16), None, None),
    ((1, 1, 32, 32), (48, 48), None, None),
    ((1, 1, 32, 32), (96, 64), None, None),
    ((1, 1, 32, 32), (64, 96), None, None),
]

SHAPE_WITH_SCALES = [
    ((1, 1, 32, 32), (64, 64), 2.0, 2.0),
    ((1, 1, 32, 32), (16, 16), 0.5, 0.5),
    ((1, 1, 32, 32), (128, 128), 4.0, 4.0),
    ((1, 1, 32, 32), (48, 48), 1.5, 1.5),
    ((1, 1, 32, 32), (80, 80), 2.5, 2.5),
    ((1, 1, 32, 32), (24, 24), 0.75, 0.75),
    ((1, 1, 32, 32), (64, 96), 2.0, 3.0),
    ((1, 1, 32, 32), (48, 64), 1.5, 2.0),
    ((1, 1, 32, 32), (16, 24), 0.5, 0.75),
    ((1, 1, 32, 32), (1, 1), 0.03125, 0.03125),
    ((1, 1, 1, 1), (32, 32), 32.0, 32.0),
]

SHAPE_MIXED_SCALES = [
    ((1, 1, 32, 32), (64, 64), 2.0, None),
    ((1, 1, 32, 32), (64, 64), None, 2.0),
    ((1, 1, 32, 32), (48, 64), 1.5, None),
    ((1, 1, 32, 32), (64, 48), None, 1.5),
]

SHAPE_UPSAMPLE = [
    ((1, 1, 16, 16), (32, 32)),
    ((1, 1, 16, 16), (48, 48)),
    ((1, 1, 16, 16), (64, 64)),
    ((1, 1, 32, 32), (96, 96)),
    ((1, 1, 64, 64), (128, 128)),
    ((1, 1, 128, 128), (256, 256)),
]

SHAPE_DOWNSAMPLE = [
    ((1, 1, 32, 32), (16, 16)),
    ((1, 1, 64, 64), (32, 32)),
    ((1, 1, 128, 128), (64, 64)),
    ((1, 1, 256, 256), (128, 128)),
    ((1, 1, 32, 32), (8, 8)),
    ((1, 1, 64, 64), (16, 16)),
]

SHAPE_SAME_SIZE = [
    ((1, 1, 8, 8), (8, 8)),
    ((1, 1, 32, 32), (32, 32)),
    ((1, 1, 64, 64), (64, 64)),
    ((1, 1, 128, 128), (128, 128)),
    ((4, 3, 32, 32), (32, 32)),
    ((8, 16, 64, 64), (64, 64)),
]

SHAPE_NON_INTEGER = [
    ((1, 1, 32, 32), (48, 48)),
    ((1, 1, 32, 32), (80, 80)),
    ((1, 1, 32, 32), (24, 24)),
    ((1, 1, 64, 64), (96, 96)),
    ((1, 1, 64, 64), (40, 40)),
]

SHAPE_ASYMMETRIC = [
    ((1, 1, 32, 32), (64, 32)),
    ((1, 1, 32, 32), (32, 64)),
    ((1, 1, 32, 32), (64, 96)),
    ((1, 1, 64, 64), (128, 32)),
    ((1, 1, 64, 64), (32, 128)),
]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_SMALL)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_small(shape, output_size, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_MEDIUM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_medium(shape, output_size, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_LARGE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_large(shape, output_size, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_DIMENSIONS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_dimensions(shape, output_size, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size, scales_h, scales_w", SHAPE_OUTPUT_SIZE_ONLY)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_output_size_only(shape, output_size, scales_h, scales_w, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(
        inp, output_size=output_size, scales_h=scales_h, scales_w=scales_w
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size, scales_h, scales_w", SHAPE_WITH_SCALES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_with_scales(shape, output_size, scales_h, scales_w, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(
        inp, output_size=output_size, scales_h=scales_h, scales_w=scales_w
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size, scales_h, scales_w", SHAPE_MIXED_SCALES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_mixed_scales(shape, output_size, scales_h, scales_w, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(
        inp, output_size=output_size, scales_h=scales_h, scales_w=scales_w
    )
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_UPSAMPLE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_upsample(shape, output_size, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_DOWNSAMPLE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_downsample(shape, output_size, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_SAME_SIZE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_same_size(shape, output_size, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_NON_INTEGER)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_non_integer(shape, output_size, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("shape, output_size", SHAPE_ASYMMETRIC)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_asymmetric(shape, output_size, dtype):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = to_reference(inp, True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_inp, output_size=output_size).to(dtype)

    res_out = flag_gems.upsample_nearest2d(inp, output_size=output_size)
    gems_assert_close(res_out, ref_out, dtype)

    if flag_gems.vendor_name == "mthreads" and dtype == torch.float16:
        del os.environ["MUSA_ENABLE_SQMMA"]
