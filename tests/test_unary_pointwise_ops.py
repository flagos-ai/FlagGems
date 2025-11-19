import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    BOOL_TYPES,
    COMPLEX_DTYPES,
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
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.abs(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.abs(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.abs_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_abs_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone())

    ref_out = torch.abs_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.abs_(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.angle
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", COMPLEX_DTYPES + FLOAT_DTYPES + ALL_INT_DTYPES + BOOL_TYPES
)
def test_accuracy_angle(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    elif dtype in ALL_INT_DTYPES:
        inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    elif dtype in COMPLEX_DTYPES + FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = to_reference(inp)
    try:
        ref_out = torch.angle(ref_inp)
    except RuntimeError as e:
        if "angle_cpu" in str(e) and "ComplexHalf" in str(e):
            pytest.skip("Skipping angle ComplexHalf for unsupported dtype on CPU")
        elif "angle_cuda" in str(e) and "Half" in str(e):
            pytest.skip("Skipping angle Half for unsupported dtype on GPU")
        elif "angle_cuda" in str(e) and "BFloat16" in str(e):
            pytest.skip("Skipping angle BFloat16 for unsupported dtype on GPU")
        else:
            raise
    ref_out = torch.angle(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.angle(inp)
    dtype_out = res_out.dtype
    gems_assert_close(res_out, ref_out, dtype_out)


BITWISE_SHAPES = [
    ((512, 1024), (512, 1024)),
    ((256, 512), (1, 512)),
    ((256, 512), (256, 1)),
    ((1, 512), (256, 512)),
    ((256, 1), (256, 512)),
    ((1024,), ()),
    ((), (1024,)),
]


@pytest.mark.bitwise_left_shift
@pytest.mark.parametrize("shapes", BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + [torch.uint8])
def test_accuracy_bitwise_left_shift(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device=flag_gems.device)
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = to_reference(res_a)
    ref_b = to_reference(res_b)

    ref_out = torch.bitwise_left_shift(ref_a, ref_b)
    with flag_gems.use_gems():
        res_out = torch.bitwise_left_shift(res_a, res_b)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.bitwise_right_shift
@pytest.mark.parametrize("shapes", BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + [torch.uint8])
def test_accuracy_bitwise_right_shift(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device=flag_gems.device)
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = to_reference(res_a)
    ref_b = to_reference(res_b)

    ref_out = torch.bitwise_right_shift(ref_a, ref_b)
    with flag_gems.use_gems():
        res_out = torch.bitwise_right_shift(res_a, res_b)
    gems_assert_close(res_out, ref_out, dtype)


INPLACE_BITWISE_SHAPES = [
    ((512, 1024), (512, 1024)),
    ((256, 512), (1, 512)),
    ((256, 512), (256, 1)),
    ((1024,), ()),
]


@pytest.mark.bitwise_left_shift
@pytest.mark.parametrize("shapes", INPLACE_BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + [torch.uint8])
def test_accuracy_bitwise_left_shift_(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device=flag_gems.device)
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = to_reference(res_a.clone())
    ref_b = to_reference(res_b)

    ref_a.bitwise_left_shift_(ref_b)
    with flag_gems.use_gems():
        res_a.bitwise_left_shift_(res_b)
    gems_assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_right_shift
@pytest.mark.parametrize("shapes", INPLACE_BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_INT_DTYPES + [torch.uint8])
def test_accuracy_bitwise_right_shift_(shapes, dtype):
    shape_a, shape_b = shapes
    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device=flag_gems.device)
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = to_reference(res_a.clone())
    ref_b = to_reference(res_b)

    ref_a.bitwise_right_shift_(ref_b)
    with flag_gems.use_gems():
        res_a.bitwise_right_shift_(res_b)
    gems_assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_not
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwisenot(shape, dtype):
    if dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.bitwise_not(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.bitwise_not(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.bitwise_not_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_accuracy_bitwisenot_(shape, dtype):
    if dtype in BOOL_TYPES:
        res_inp = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    else:
        res_inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp = to_reference(res_inp.clone())

    ref_out = ref_inp.bitwise_not_()  # NOTE: there is no torch.bitwse_not_
    with flag_gems.use_gems():
        res_out = res_inp.bitwise_not_()

    gems_assert_equal(res_out, ref_out)


@pytest.mark.cos
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.cos(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.cos_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_cos_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.cos_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cos_(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.exp(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.exp(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.exp_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.exp_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.exp_(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.exp2
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp2(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    if flag_gems.vendor_name == "kunlunxin":
        ref_out = torch.exp2(ref_inp.cpu()).to(flag_gems.device)
    else:
        ref_out = torch.exp2(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.exp2(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.exp2_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_exp2_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    if flag_gems.vendor_name == "kunlunxin":
        ref_out = torch.exp2_(ref_inp.cpu()).to(flag_gems.device)
    else:
        ref_out = torch.exp2_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.exp2_(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.gelu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.nn.functional.gelu(ref_inp, approximate=approximate)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.gelu(res_inp, approximate=approximate)

    atol = 1e-4
    if flag_gems.vendor_name == "aipu" and dtype == torch.float16:
        atol = 1e-3
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.gelu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu_backward(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn_like(res_inp)

    ref_inp = to_reference(res_inp, True)
    ref_out = to_reference(res_out, True)

    ref_in_grad = torch.ops.aten.gelu_backward(
        ref_out, ref_inp, approximate=approximate
    )
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.gelu_backward(
            res_out, res_inp, approximate=approximate
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.gelu_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_accuracy_gelu_(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp.clone(), True)

    ref_out = torch.ops.aten.gelu_.default(ref_inp, approximate=approximate)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.gelu_.default(res_inp, approximate=approximate)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.glu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_glu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    for dim in range(len(shape)):
        if shape[dim] % 2 != 0:
            continue
        ref_out = torch.nn.functional.glu(ref_inp, dim=dim)
        with flag_gems.use_gems():
            res_out = torch.nn.functional.glu(res_inp, dim=dim)
        gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.glu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_glu_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    for dim in range(len(shape)):
        if shape[dim] == 0 or shape[dim] % 2 != 0:
            continue
        out_shape = list(shape)
        out_shape[dim] //= 2
        res_out = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
        ref_out = to_reference(res_out, True)

        ref_in_grad = torch.ops.aten.glu_backward(ref_out, ref_inp, dim=dim)
        with flag_gems.use_gems():
            res_in_grad = torch.ops.aten.glu_backward(res_out, res_inp, dim=dim)

        gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.isinf
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_isinf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.neg(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.neg(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.inplace
@pytest.mark.neg_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_neg_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone())

    ref_out = torch.neg_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.neg_(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.reciprocal
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.reciprocal(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.reciprocal_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_reciprocal_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.reciprocal_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal_(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.elu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_elu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    ref_inp = to_reference(inp, True)
    ref_out = torch.nn.functional.elu(ref_inp, alpha)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.elu(inp, alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.elu_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_elu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    res_inp = inp.clone().to(flag_gems.device)
    inp_clone = inp.clone()
    ref_inp = to_reference(inp_clone, True)
    torch.nn.functional.elu_(ref_inp, alpha)

    with flag_gems.use_gems():
        torch.nn.functional.elu_(res_inp, alpha)

    gems_assert_close(res_inp, ref_inp, dtype)


@pytest.mark.elu_backward
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("is_result", [True, False])
def test_accuracy_elu_backward(shape, dtype, is_result):
    alpha = torch.rand(1).item()
    scale = 1.0
    input_scale = 1.0

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad_out = torch.randn_like(res_inp)

    if is_result:
        res_self_or_result = torch.ops.aten.elu(res_inp, alpha, scale, input_scale)
    else:
        res_self_or_result = res_inp

    ref_grad_out = to_reference(res_grad_out, True)
    ref_self_or_result = to_reference(res_self_or_result, True)

    ref_in_grad = torch.ops.aten.elu_backward(
        ref_grad_out, alpha, scale, input_scale, is_result, ref_self_or_result
    )

    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.elu_backward(
            res_grad_out, alpha, scale, input_scale, is_result, res_self_or_result
        )

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.celu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_celu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    ref_inp = to_reference(inp, True)
    ref_out = torch.nn.functional.celu(ref_inp, alpha)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.celu(inp, alpha)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.celu_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_celu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    alpha = torch.rand(1).item()

    res_inp = inp.clone().to(flag_gems.device)
    inp_clone = inp.clone()
    ref_inp = to_reference(inp_clone, True)
    torch.nn.functional.celu_(ref_inp, alpha)

    with flag_gems.use_gems():
        torch.nn.functional.celu_(res_inp, alpha)

    gems_assert_close(res_inp, ref_inp, dtype)


@pytest.mark.relu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_relu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.relu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.relu(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.relu_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_relu_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp.clone(), True)

    ref_out = torch.relu_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.relu_(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.softplus
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_softplus(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    beta = torch.rand(1).item()
    threshold = torch.rand(1).item() * 40.0
    ref_inp = to_reference(inp, True)
    ref_out = torch.nn.functional.softplus(ref_inp, beta=beta, threshold=threshold)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softplus(inp, beta=beta, threshold=threshold)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rsqrt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.rsqrt(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.rsqrt(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.inplace
@pytest.mark.rsqrt_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_rsqrt_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.rsqrt_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.rsqrt_(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.sigmoid
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sigmoid(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.sigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sigmoid(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sigmoid
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sigmoid_backward(shape, dtype):
    res_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_out)

    ref_out = to_reference(res_out, True)
    ref_grad = to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.sigmoid_backward(ref_grad, ref_out)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.sigmoid_backward(res_grad, res_out)

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.inplace
@pytest.mark.sigmoid_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sigmoid_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp.clone(), True)

    ref_out = torch.sigmoid_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sigmoid_(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


SPECIAL_VALUES = [float("-inf"), float("inf"), -300]


@pytest.mark.log_sigmoid
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log_sigmoid(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if len(shape) == 1:
        special_inputs = torch.tensor(
            SPECIAL_VALUES, dtype=dtype, device=flag_gems.device
        )
        inp = torch.cat((inp, special_inputs))
    ref_inp = to_reference(inp, True)

    ref_out = torch.nn.functional.logsigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.logsigmoid(inp)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.silu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.nn.functional.silu(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.silu
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)

    ref_inp = to_reference(res_inp, True)
    ref_grad = to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.silu_backward(ref_grad, ref_inp)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.silu_backward(res_grad, res_inp)

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.inplace
@pytest.mark.silu_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_silu_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp.clone(), True)

    ref_out = torch.nn.functional.silu(ref_inp, inplace=True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(res_inp, inplace=True)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.sin
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sin(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.sin_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.sin_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sin_(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tanh
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tanh(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.tanh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.tanh(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.tanh
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tanh_backward(shape, dtype):
    res_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_out)

    ref_out = to_reference(res_out, True)
    ref_grad = to_reference(res_grad, True)

    ref_in_grad = torch.ops.aten.tanh_backward(ref_grad, ref_out)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.tanh_backward(res_grad, res_out)

    gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.inplace
@pytest.mark.tanh_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tanh_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp.clone(), True)

    ref_out = torch.tanh_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.tanh_(res_inp)

    gems_assert_close(res_out, ref_out, dtype)


SHAPE_DIAGONAL = list(zip(POINTWISE_SHAPES, [-2, -2, -1, 0, 1, 3]))


@pytest.mark.triu
@pytest.mark.parametrize("shape, diagonal", SHAPE_DIAGONAL)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.erf(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.erf(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.erf_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_erf_(shape, dtype):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone())

    ref_out = torch.erf_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.erf_(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.isfinite
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES)
def test_accuracy_isfinite(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
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
        inp = torch.randn(shape_dialted, dtype=dtype, device=flag_gems.device)[::2, ::2]
    else:
        inp = torch.randint(-1000, 1000, shape_dialted, device=flag_gems.device).to(
            dtype
        )[::2, ::2]
    ref_inp = to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.flip(inp, dims)
    ref_out = torch.flip(ref_inp, dims)
    gems_assert_equal(res_out, ref_out)


TILE_DIMS = [(0,), (2,), (2, 0), (0, 2), (2, 2), (2, 2, 2), (2, 2, 2, 2)]


@pytest.mark.tile
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dims", TILE_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_tile(shape, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
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
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in ALL_INT_DTYPES:
        inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    elif dtype in BOOL_TYPES:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)

    ref_inp = to_reference(inp)
    ref_out = torch.logical_not(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.logical_not(inp)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.log
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_log(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = to_reference(inp, True)
    ref_out = torch.log(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_to_dtype(shape, dtype):
    x = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = ref_x.to(dtype)
    with flag_gems.use_gems():
        out = x.to(dtype)
    gems_assert_equal(out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("target_dtype", ALL_FLOAT_DTYPES)
def test_accuracy_to_copy_dtype_cast(shape, target_dtype):
    src_dtype = torch.float32 if target_dtype != torch.float32 else torch.float16
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=target_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=target_dtype)
    gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.contiguous_format],
)
def test_accuracy_to_copy_preserve_strides(memory_format):
    base = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    x = base.transpose(0, 1)[::2]
    ref_x = to_reference(x)
    ref_out = torch.ops.aten._to_copy(
        ref_x,
        dtype=ref_x.dtype,
        memory_format=memory_format,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(
            x,
            dtype=x.dtype,
            memory_format=memory_format,
        )
    gems_assert_equal(res_out, ref_out)
    if memory_format is torch.preserve_format:
        assert res_out.stride() == ref_out.stride()
    else:
        assert res_out.is_contiguous()


@pytest.mark.sqrt
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES)
def test_accuracy_sqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.sqrt(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sqrt(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.sqrt_
@pytest.mark.inplace
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES)
def test_accuracy_sqrt_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone(), True)

    ref_out = torch.sqrt_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sqrt_(inp)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.atan
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_atan(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp, True)

    ref_out = torch.atan(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.atan(res_inp)
    ref_out = ref_out.to(res_out.dtype)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.atan_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_atan_(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(res_inp.clone(), True)

    ref_out = torch.atan_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.atan_(res_inp)

    ref_out = ref_out.to(res_out.dtype)
    gems_assert_close(res_out, ref_out, dtype)
