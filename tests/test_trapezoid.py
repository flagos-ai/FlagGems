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
def test_trapezoid_x_1d(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    n = shape[-1]
    res_x = torch.sort(torch.randn(n, dtype=dtype, device=flag_gems.device))[0]
    ref_inp = utils.to_reference(res_inp, upcast=True)
    ref_x = utils.to_reference(res_x, upcast=True)

    ref_out = torch.trapezoid(ref_inp, ref_x)
    res_out = flag_gems.trapezoid_x(res_inp, res_x)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=n)


@pytest.mark.trapezoid
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


@pytest.mark.trapezoid
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


@pytest.mark.trapezoid
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


@pytest.mark.trapezoid
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


@pytest.mark.trapezoid
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


@pytest.mark.trapezoid
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


@pytest.mark.trapezoid
def test_trapezoid_x_invalid_dim():
    # Out-of-range dims raise IndexError with ATen's wording, checked on both
    # the empty-dim early return and the kernel path.
    y = torch.randn(2, 3, device=flag_gems.device)
    x = torch.randn(3, device=flag_gems.device)
    for bad_dim in (5, -3):
        with pytest.raises(IndexError, match="Dimension out of range"):
            flag_gems.trapezoid_x(y, x, dim=bad_dim)
    with pytest.raises(IndexError):
        torch.trapezoid(y, x, dim=5)


@pytest.mark.trapezoid
def test_trapezoid_x_empty_dim_skips_length_check():
    # ATen does not reject a mismatched 1-D x when the integration dim is empty
    # (the integral is zero there regardless), so the length check must not
    # fire before that early return.
    y = torch.randn(10, 0, device=flag_gems.device)
    x = torch.randn(5, device=flag_gems.device)

    res_out = flag_gems.trapezoid_x(y, x, dim=-1)
    ref_out = utils.to_reference(torch.zeros(10, device=flag_gems.device), upcast=False)
    utils.gems_assert_close(res_out, ref_out, res_out.dtype)


@pytest.mark.trapezoid
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


@pytest.mark.trapezoid
def test_trapezoid_x_mixed_dtype():
    # int y + float x promotes to the float x dtype (ATen result_type).
    y = torch.randint(-100, 100, (4, 5), dtype=torch.int32, device=flag_gems.device)
    x = torch.sort(
        torch.randn(4, 5, dtype=torch.float16, device=flag_gems.device), dim=-1
    )[0]
    ref_y = utils.to_reference(y)
    ref_x = utils.to_reference(x, upcast=True)

    ref_out = torch.trapezoid(ref_y, ref_x)
    res_out = flag_gems.trapezoid_x(y, x)

    utils.gems_assert_close(res_out, ref_out, torch.float16, reduce_dim=5)

    # float16 y + float32 x promotes to float32.
    y = torch.randn(4, 5, dtype=torch.float16, device=flag_gems.device)
    x = torch.sort(
        torch.randn(4, 5, dtype=torch.float32, device=flag_gems.device), dim=-1
    )[0]
    ref_y = utils.to_reference(y, upcast=True)
    ref_x = utils.to_reference(x, upcast=True)

    ref_out = torch.trapezoid(ref_y, ref_x)
    res_out = flag_gems.trapezoid_x(y, x)

    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=5)

    # int y + int x promotes to float32.
    y = torch.randint(-100, 100, (4, 5), dtype=torch.int32, device=flag_gems.device)
    x = torch.sort(
        torch.randint(-100, 100, (4, 5), dtype=torch.int32, device=flag_gems.device),
        dim=-1,
    )[0]
    ref_y = utils.to_reference(y)
    ref_x = utils.to_reference(x)

    ref_out = torch.trapezoid(ref_y, ref_x)
    res_out = flag_gems.trapezoid_x(y, x)

    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=5)


@pytest.mark.trapezoid
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_trapezoid_x_complex(dtype):
    # Complex inputs follow ATen's contract: same-rank broadcastable shapes,
    # forward in the complex dtype, and Wirtinger-convention backward
    # (grad = conj(d out / d input)). The device path decomposes the complex
    # arithmetic through real-view reductions.
    for dim, y_shape, x_len in [(-1, (3, 5), 5), (0, (3, 5), 3)]:
        y = torch.randn(
            y_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        x = torch.randn(x_len, dtype=dtype, device=flag_gems.device, requires_grad=True)
        ref_y = utils.to_reference(y.detach().clone(), upcast=False).requires_grad_()
        ref_x = utils.to_reference(x.detach().clone(), upcast=False).requires_grad_()

        ref_out = torch.trapezoid(ref_y, ref_x, dim=dim)
        res_out = flag_gems.trapezoid_x(y, x, dim=dim)

        assert res_out.dtype == dtype
        utils.gems_assert_close(res_out, ref_out, dtype)

        # Same upstream gradient values on both devices: in --ref=cpu mode the
        # reference lives on cpu while the gems result stays on the device.
        grad = torch.randn(res_out.shape, dtype=dtype, device=res_out.device)
        res_out.backward(grad.clone())
        ref_out.backward(grad.to(ref_out.device).clone())
        utils.gems_assert_close(y.grad, ref_y.grad, dtype)
        utils.gems_assert_close(x.grad, ref_x.grad, dtype)


@pytest.mark.trapezoid
def test_trapezoid_x_complex_mixed():
    # Real y + complex x promotes to the complex dtype (ATen result_type).
    y = torch.randn(3, 5, device=flag_gems.device)
    x = torch.randn(5, dtype=torch.complex64, device=flag_gems.device)
    ref_y = utils.to_reference(y)
    ref_x = utils.to_reference(x)

    ref_out = torch.trapezoid(ref_y, ref_x)
    res_out = flag_gems.trapezoid_x(y, x)
    assert res_out.dtype == torch.complex64
    utils.gems_assert_close(res_out, ref_out, torch.complex64)


@pytest.mark.trapezoid
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_x_y_broadcasting(dtype):
    # y broadcasts to x (y's leading dim is 1, x's is larger).
    y_shape = (1, 5, 6)
    x_shape = (4, 5, 6)
    res_y = torch.randn(y_shape, dtype=dtype, device=flag_gems.device)
    res_x = torch.sort(
        torch.randn(x_shape, dtype=dtype, device=flag_gems.device), dim=-1
    )[0]
    ref_y = utils.to_reference(res_y, upcast=True)
    ref_x = utils.to_reference(res_x, upcast=True)

    ref_out = torch.trapezoid(ref_y, ref_x)
    res_out = flag_gems.trapezoid_x(res_y, res_x)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=y_shape[-1])


@pytest.mark.trapezoid
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapezoid_x_pair_broadcasting(dtype):
    # y has 2 sample points (pair length 1) and x has 5 (pair length 4); ATen
    # broadcasts the pair axes to length 4. Exercise both directions.
    y = torch.randn(2, 2, dtype=dtype, device=flag_gems.device)
    x = torch.sort(torch.randn(2, 5, dtype=dtype, device=flag_gems.device), dim=-1)[0]
    ref_y = utils.to_reference(y, upcast=True)
    ref_x = utils.to_reference(x, upcast=True)

    ref_out = torch.trapezoid(ref_y, ref_x, dim=1)
    res_out = flag_gems.trapezoid_x(y, x, dim=1)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=4)

    y = torch.randn(2, 5, dtype=dtype, device=flag_gems.device)
    x = torch.sort(torch.randn(2, 2, dtype=dtype, device=flag_gems.device), dim=-1)[0]
    ref_y = utils.to_reference(y, upcast=True)
    ref_x = utils.to_reference(x, upcast=True)

    ref_out = torch.trapezoid(ref_y, ref_x, dim=1)
    res_out = flag_gems.trapezoid_x(y, x, dim=1)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=4)


@pytest.mark.trapezoid
def test_trapezoid_x_backward():
    y = torch.randn(
        (3, 5), dtype=torch.float64, device=flag_gems.device, requires_grad=True
    )
    x = torch.sort(torch.randn(5, dtype=torch.float64, device=flag_gems.device))[0]
    x.requires_grad_(True)
    ref_y = utils.to_reference(y.detach().clone(), upcast=False).requires_grad_()
    ref_x = utils.to_reference(x.detach().clone(), upcast=False).requires_grad_()
    torch.trapezoid(ref_y, ref_x).sum().backward()
    flag_gems.trapezoid_x(y, x).sum().backward()

    utils.gems_assert_close(y.grad, ref_y.grad, torch.float64)
    utils.gems_assert_close(x.grad, ref_x.grad, torch.float64)


@pytest.mark.trapezoid
def test_trapezoid_x_backward_single_point():
    # N == 1 (single sample point) spans no interval: ATen returns zero gradients
    # for both y and x. Verify parity with the reference (N == 1 keeps grad_fn).
    y = torch.randn(
        (3, 1), dtype=torch.float64, device=flag_gems.device, requires_grad=True
    )
    x = torch.sort(torch.randn(1, dtype=torch.float64, device=flag_gems.device))[0]
    x.requires_grad_(True)
    ref_y = utils.to_reference(y.detach().clone(), upcast=False).requires_grad_()
    ref_x = utils.to_reference(x.detach().clone(), upcast=False).requires_grad_()

    torch.trapezoid(ref_y, ref_x, dim=1).sum().backward()
    flag_gems.trapezoid_x(y, x, dim=1).sum().backward()

    utils.gems_assert_close(y.grad, ref_y.grad, torch.float64)
    utils.gems_assert_close(x.grad, ref_x.grad, torch.float64)


@pytest.mark.trapezoid
def test_trapezoid_x_backward_empty():
    # N == 0 (empty pair axis) spans no interval. ATen's forward detaches the
    # empty result from the graph, so there is no reference backward to compare;
    # verify the gems backward runs without a negative-length narrow and yields
    # zero gradients for both inputs.
    y = torch.randn(
        (3, 0), dtype=torch.float64, device=flag_gems.device, requires_grad=True
    )
    x = torch.randn(0, dtype=torch.float64, device=flag_gems.device, requires_grad=True)

    flag_gems.trapezoid_x(y, x, dim=1).sum().backward()

    assert torch.all(y.grad == 0)
    assert torch.all(x.grad == 0)
