import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
    unsqueeze_tensor,
    unsqueeze_tuple,
)


@pytest.mark.abs
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.abs(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.abs(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_not
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwisenot(shape, dtype):
    if dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cuda"
        )
    ref_inp = to_reference(inp)

    ref_out = torch.bitwise_not(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.bitwise_not(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.cos
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.exp(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.exp(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.gelu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu(shape, dtype, approximate):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.gelu(ref_inp, approximate=approximate)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.gelu(inp, approximate=approximate)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.isinf
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    ref_inp = to_reference(inp)

    ref_out = torch.isinf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isinf(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.isnan
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isnan(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp = torch.masked_fill(inp, inp > 1.0, float("nan"))
    ref_inp = to_reference(inp)

    ref_out = torch.isnan(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isnan(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.neg
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_neg(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.neg(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.neg(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.reciprocal
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.reciprocal(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.relu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.relu(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.rsqrt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.rsqrt(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.rsqrt(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.sigmoid
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sigmoid(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sigmoid(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.silu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.sin
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.sin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tanh
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tanh(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    ref_inp = to_reference(inp, True)

    ref_out = torch.tanh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.tanh(inp)

    gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = to_reference(out_grad, True)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    gems_assert_close(res_in_grad, ref_in_grad, dtype)


SHAPE_DIAGONAL = list(zip(POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))


@pytest.mark.triu
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp = unsqueeze_tensor(inp, 2)
    ref_inp = to_reference(inp)

    ref_out = torch.triu(ref_inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.triu(inp, diagonal)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.erf
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_erf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.erf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.erf(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.isfinite
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES)
def test_accuracy_isfinite(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    inp = torch.masked_fill(inp, inp > 1.0, float("inf"))
    inp = torch.masked_fill(inp, inp < -1.0, float("-inf"))
    inp = torch.masked_fill(inp, (inp > -0.1) & (inp < 0.1), float("nan"))
    ref_inp = to_reference(inp)

    ref_out = torch.isfinite(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.isfinite(inp)
    gems_assert_equal(res_out, ref_out)


def get_max_ndim(shape, dims):
    max_ndim = max(len(shape), len(dims))
    for dim in dims:
        dim = dim + 1 if dim >= 0 else -dim
        if dim > max_ndim:
            max_ndim = dim
    return max_ndim


FLIP_DIMS = [(0,), (-2,), (2,), (0, 2), (2, 1), (0, -1, 1)]


@pytest.mark.flip
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dims", FLIP_DIMS)
def test_accuracy_flip_general(shape, dtype, dims):
    if dtype in ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cuda")
    else:
        inp = torch.randint(-1000, 1000, shape, device="cuda").to(dtype)
    max_ndim = get_max_ndim(shape, dims)
    inp = unsqueeze_tensor(inp, max_ndim)
    ref_inp = to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.flip(inp, dims)
    ref_out = torch.flip(ref_inp, dims)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.flip
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
@pytest.mark.parametrize("dims", FLIP_DIMS)
def test_accuracy_flip_with_non_dense_input(shape, dtype, dims):
    max_ndim = get_max_ndim(shape, dims)
    shape = unsqueeze_tuple(shape, max(max_ndim, 2))

    shape_dialted = tuple(item * 2 for item in shape)
    if dtype in ALL_FLOAT_DTYPES:
        inp = torch.randn(shape_dialted, dtype=dtype, device="cuda")[::2, ::2]
    else:
        inp = torch.randint(-1000, 1000, shape_dialted, device="cuda").to(dtype)[
            ::2, ::2
        ]
    ref_inp = to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.flip(inp, dims)
    ref_out = torch.flip(ref_inp, dims)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize(
    "value",
    [torch.tensor(1024, device="cuda"), torch.scalar_tensor(1024, device="cuda"), 1024],
)
def test_accuracy_masked_fill(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device="cuda")
    mask = torch.randn(shape, dtype=dtype, device="cuda") < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    if torch.is_tensor(value):
        ref_out = torch.masked_fill(ref_inp, ref_mask, to_reference(value))
    else:
        ref_out = torch.masked_fill(ref_inp, ref_mask, value)
    with flag_gems.use_gems():
        res_out = torch.masked_fill(inp, mask, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.masked_fill
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize(
    "value",
    [torch.tensor(1024, device="cuda"), torch.scalar_tensor(1024, device="cuda"), 1024],
)
def test_accuracy_masked_fill_(shape, dtype, threshold, value):
    inp = torch.zeros(shape, dtype=dtype, device="cuda")
    mask = torch.randn(shape, dtype=dtype, device="cuda") < threshold

    ref_inp = to_reference(inp)
    ref_mask = to_reference(mask)
    if torch.is_tensor(value):
        ref_inp.masked_fill_(ref_mask, to_reference(value))
    else:
        ref_inp.masked_fill_(ref_mask, value)
    with flag_gems.use_gems():
        inp.masked_fill_(mask, value)

    gems_assert_equal(inp, ref_inp)


TILE_DIMS = [(0,), (2,), (2, 0), (0, 2), (2, 2), (2, 2, 2), (2, 2, 2, 2)]


@pytest.mark.tile
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dims", TILE_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tile(shape, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)

    ref_out = torch.tile(ref_inp, dims)
    with flag_gems.use_gems():
        res_out = torch.tile(inp, dims)

    gems_assert_close(res_out, ref_out, dtype)


REPEAT_SIZES = [(2, 3, 4, 5), (5, 0, 4)]


@pytest.mark.repeat
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("sizes", REPEAT_SIZES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_repeat(shape, sizes, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp)
    sizes = unsqueeze_tuple(sizes, inp.ndim)

    ref_out = ref_inp.repeat(*sizes)
    with flag_gems.use_gems():
        res_out = inp.repeat(*sizes)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.logical_not
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES + BOOL_TYPES)
def test_accuracy_logical_not(shape, dtype):
    if dtype in ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cuda")
    elif dtype in ALL_INT_DTYPES:
        inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cuda")
    elif dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cuda")

    ref_inp = to_reference(inp)
    ref_out = torch.logical_not(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.logical_not(inp)

    gems_assert_equal(res_out, ref_out)
