import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# trapezoid reduces along a dim, so shapes need at least one dim with length > 1
# for the pairwise sum to be meaningful. Scalar/1-length shapes are also tested
# to exercise the degenerate (empty pair) case.
TRAPEZOID_SHAPES = [
    (1,),
    (1024,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
]


@pytest.mark.trapezoid
@pytest.mark.parametrize("shape", TRAPEZOID_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_dx_default(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=True)

    ref_out = torch.trapezoid(ref_inp)
    res_out = flag_gems.trapezoid(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[-1])


@pytest.mark.trapezoid
@pytest.mark.parametrize("dx", [0.5, 2.0, 3.5])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_dx_value(dx, dtype):
    # 3-D mid-size shape: exercises reduction over the last dim (15) with a
    # non-trivial outer batch (20 x 320) while staying small enough for fp16.
    shape = (20, 320, 15)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=True)

    ref_out = torch.trapezoid(ref_inp, dx=dx)
    res_out = flag_gems.trapezoid(res_inp, dx=dx)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[-1])


@pytest.mark.trapezoid
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_dx_dim(dim, dtype):
    # 3-D shape with distinct extents per axis so reducing over dim 0/1/2/-1
    # each covers a different length and validates arbitrary-dim reduction.
    shape = (20, 32, 15)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=True)

    ref_out = torch.trapezoid(ref_inp, dx=2.0, dim=dim)
    res_out = flag_gems.trapezoid(res_inp, dx=2.0, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.trapezoid
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_trapezoid_dx_int(dtype):
    # 3-D mid-size shape mirroring the float case, sized for integer inputs
    # so the reduction over the last dim (15) accumulates without overflow.
    shape = (20, 320, 15)
    res_inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.trapezoid(ref_inp, dx=2.0)
    res_out = flag_gems.trapezoid(res_inp, dx=2.0)

    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=shape[-1])


@pytest.mark.trapezoid
@pytest.mark.parametrize("dtype", [torch.float64])
def test_trapezoid_dx_fp64(dtype):
    # Test FP64 precision is preserved
    shape = (20, 32, 15)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=False)

    ref_out = torch.trapezoid(ref_inp, dx=2.0)
    res_out = flag_gems.trapezoid(res_inp, dx=2.0)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[-1])


@pytest.mark.trapezoid
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_dx_non_contiguous(dtype):
    # Test non-contiguous input
    shape = (20, 32, 15)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(res_inp, upcast=True)

    ref_out = torch.trapezoid(ref_inp, dx=2.0)
    res_out = flag_gems.trapezoid(res_inp, dx=2.0)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[0])


@pytest.mark.trapezoid
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_dx_zero_size(dtype):
    # Test zero-sized dimension
    shape = (0, 10)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, upcast=True)

    ref_out = torch.trapezoid(ref_inp, dx=2.0)
    res_out = flag_gems.trapezoid(res_inp, dx=2.0)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.trapezoid_x
@pytest.mark.parametrize("shape", TRAPEZOID_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_x_1d(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = shape[-1]
    res_x = torch.sort(torch.randn(n, dtype=dtype, device=flag_gems.device))[0]
    ref_inp = utils.to_reference(res_inp, upcast=True)
    ref_x = utils.to_reference(res_x, upcast=True)

    ref_out = torch.trapezoid(ref_inp, ref_x)
    res_out = flag_gems.trapezoid_x(res_inp, res_x)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.trapezoid_x
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_x_same_shape(dtype):
    # 3-D shape where x matches y exactly, covering the elementwise-spacing path
    # (x broadcast identical to the input) with a small fp16-safe extent.
    shape = (20, 32, 15)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_x = torch.sort(
        torch.randn(shape, dtype=dtype, device=flag_gems.device), dim=-1
    )[0]
    ref_inp = utils.to_reference(res_inp, upcast=True)
    ref_x = utils.to_reference(res_x, upcast=True)

    ref_out = torch.trapezoid(ref_inp, ref_x)
    res_out = flag_gems.trapezoid_x(res_inp, res_x)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[-1])


@pytest.mark.trapezoid_x
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_x_dim(dim, dtype):
    # 3-D shape with distinct extents per axis so the 1-D x-spacing variant is
    # validated when reducing over each of dim 0/1/2/-1.
    shape = (20, 32, 15)
    n = shape[dim]
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_x = torch.sort(torch.randn(n, dtype=dtype, device=flag_gems.device))[0]
    ref_inp = utils.to_reference(res_inp, upcast=True)
    ref_x = utils.to_reference(res_x, upcast=True)

    ref_out = torch.trapezoid(ref_inp, ref_x, dim=dim)
    res_out = flag_gems.trapezoid_x(res_inp, res_x, dim=dim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.trapezoid_x
@pytest.mark.parametrize("dtype", [torch.float64])
def test_trapezoid_x_fp64(dtype):
    # Test FP64 precision is preserved for trapezoid_x
    shape = (20, 32, 15)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_x = torch.sort(torch.randn(shape[-1], dtype=dtype, device=flag_gems.device))[0]
    ref_inp = utils.to_reference(res_inp, upcast=False)
    ref_x = utils.to_reference(res_x, upcast=False)

    ref_out = torch.trapezoid(ref_inp, ref_x)
    res_out = flag_gems.trapezoid_x(res_inp, res_x)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[-1])


@pytest.mark.trapezoid_x
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_x_non_contiguous(dtype):
    # Test non-contiguous y and x
    shape = (20, 32, 15)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).transpose(0, 1)
    # After transpose(0,1), shape becomes (32, 20, 15), so dim=0 needs 32 elements
    res_x = torch.sort(torch.randn(32, dtype=dtype, device=flag_gems.device))[0]
    ref_inp = utils.to_reference(res_inp, upcast=True)
    ref_x = utils.to_reference(res_x, upcast=True)

    ref_out = torch.trapezoid(ref_inp, ref_x, dim=0)
    res_out = flag_gems.trapezoid_x(res_inp, res_x, dim=0)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=32)


@pytest.mark.trapezoid_x
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_x_broadcasting(dtype):
    # Test ATen-compatible broadcasting: x broadcasts to y
    y_shape = (4, 5, 6)
    x_shape = (1, 5, 6)
    res_y = torch.randn(y_shape, dtype=dtype, device=flag_gems.device)
    res_x = torch.sort(
        torch.randn(x_shape, dtype=dtype, device=flag_gems.device), dim=-1
    )[0]
    ref_y = utils.to_reference(res_y, upcast=True)
    ref_x = utils.to_reference(res_x, upcast=True)

    ref_out = torch.trapezoid(ref_y, ref_x)
    res_out = flag_gems.trapezoid_x(res_y, res_x)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=y_shape[-1])


@pytest.mark.trapezoid_x
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_x_mismatched_length(dtype):
    # Test that mismatched x length raises error
    shape = (20, 32, 15)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Wrong length: 10 instead of 15
    res_x = torch.sort(torch.randn(10, dtype=dtype, device=flag_gems.device))[0]

    with pytest.raises(
        RuntimeError, match="There must be one `x` value for each sample point"
    ):
        flag_gems.trapezoid_x(res_inp, res_x)


@pytest.mark.trapezoid_x
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_x_zero_size(dtype):
    # Test zero-sized dimension with x
    shape = (0, 10)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_x = torch.sort(torch.randn(10, dtype=dtype, device=flag_gems.device))[0]
    ref_inp = utils.to_reference(res_inp, upcast=True)
    ref_x = utils.to_reference(res_x, upcast=True)

    ref_out = torch.trapezoid(ref_inp, ref_x)
    res_out = flag_gems.trapezoid_x(res_inp, res_x)

    utils.gems_assert_close(res_out, ref_out, dtype)
