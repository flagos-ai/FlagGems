import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SHAPE_CONV_TRANSPOSE2D = [
    ((1, 2, 5, 5), (2, 3, 3, 3), 1),
    ((2, 4, 7, 7), (4, 2, 3, 3), 2),
    ((4, 8, 8, 8), (8, 4, 2, 2), 2),
]

CONV_TRANSPOSE2D_FORWARD_DTYPES = [torch.float16, torch.float32]
if flag_gems.runtime.device.support_bf16:
    CONV_TRANSPOSE2D_FORWARD_DTYPES.append(torch.bfloat16)


def _disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _skip_cpu_bf16_reference(dtype):
    if utils.TO_CPU and dtype == torch.bfloat16:
        pytest.skip(
            "CPU bf16 conv_transpose2d uses a different accumulation path from CUDA"
        )


def _conv_transpose2d_forward_atol(dtype, reduce_dim):
    if dtype == torch.bfloat16:
        return 0.016 / reduce_dim
    if dtype == torch.float16:
        return 1e-3
    return 1e-4


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D)
@pytest.mark.parametrize(
    "stride, padding, output_padding, dilation",
    [
        (1, 0, 0, 1),
        (2, 1, 0, 1),
        ((2, 2), (1, 1), (1, 1), 1),
        ((2, 1), (1, 0), (1, 0), (2, 1)),
    ],
)
@pytest.mark.parametrize("dtype", CONV_TRANSPOSE2D_FORWARD_DTYPES)
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv_transpose2d_forward(
    shape, kernel, groups, stride, padding, output_padding, dilation, dtype, bias
):
    _skip_cpu_bf16_reference(dtype)
    _disable_tf32()
    reduce_dim = (shape[1] // groups) * kernel[2] * kernel[3]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=False)
    ref_inp = utils.to_reference(inp, False)
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    ref_weight = utils.to_reference(weight, False)

    if bias:
        bias_tensor = torch.randn(
            [kernel[1] * groups],
            dtype=dtype,
            device=flag_gems.device,
            requires_grad=False,
        )
        bias_ref = utils.to_reference(bias_tensor, False)
    else:
        bias_tensor = None
        bias_ref = None

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    ).to(dtype)

    with torch.no_grad():
        res_out = flag_gems.conv_transpose2d(
            inp,
            weight,
            bias=bias_tensor,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )

    atol = _conv_transpose2d_forward_atol(dtype, reduce_dim)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim, atol=atol)


CONV_TRANSPOSE2D_FORWARD_EDGE_CASES = [
    ((1, 1, 1, 1), (1, 1, 1, 1), 1, 1, 0, 0, 1, False, False),
    ((1, 2, 4, 4), (2, 3, 3, 3), 1, 4, 1, 0, 1, False, False),
    ((1, 3, 4, 5), (3, 2, 2, 3), 1, (3, 2), (1, 0), (2, 1), 1, True, False),
    ((2, 4, 5, 6), (4, 3, 5, 5), 1, 2, 2, 1, 1, True, True),
    ((4, 4, 64, 64), (4, 4, 5, 5), 1, 3, 2, 0, 1, False, False),
    ((1, 8, 4, 4), (8, 1, 3, 3), 4, 2, 1, 1, 1, False, False),
    (
        (1, 2, 5, 4),
        (2, 3, 3, 2),
        1,
        (1, 2),
        (0, 1),
        (0, 1),
        (2, 1),
        False,
        True,
    ),
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    (
        "shape, kernel, groups, stride, padding, output_padding, dilation, "
        "noncontig_input, noncontig_weight"
    ),
    CONV_TRANSPOSE2D_FORWARD_EDGE_CASES,
)
@pytest.mark.parametrize("dtype", CONV_TRANSPOSE2D_FORWARD_DTYPES)
def test_accuracy_conv_transpose2d_forward_edge_cases(
    shape,
    kernel,
    groups,
    stride,
    padding,
    output_padding,
    dilation,
    noncontig_input,
    noncontig_weight,
    dtype,
):
    _skip_cpu_bf16_reference(dtype)
    _disable_tf32()
    if noncontig_input:
        inp_shape = (shape[0], shape[1], shape[3], shape[2])
        inp = torch.randn(
            inp_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        ).transpose(2, 3)
    else:
        inp = torch.randn(
            shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )
    ref_inp = utils.to_reference(inp, False)

    if noncontig_weight:
        weight_shape = (kernel[0], kernel[1], kernel[3], kernel[2])
        weight = torch.randn(
            weight_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        ).transpose(2, 3)
    else:
        weight = torch.randn(
            kernel, dtype=dtype, device=flag_gems.device, requires_grad=False
        )
    ref_weight = utils.to_reference(weight, False)

    bias_tensor = torch.randn(
        [kernel[1] * groups],
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=False,
    )
    bias_ref = utils.to_reference(bias_tensor, False)
    reduce_dim = (shape[1] // groups) * kernel[2] * kernel[3]

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    ).to(dtype)

    with torch.no_grad():
        res_out = flag_gems.conv_transpose2d(
            inp,
            weight,
            bias=bias_tensor,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )

    atol = _conv_transpose2d_forward_atol(dtype, reduce_dim)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim, atol=atol)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize(
    "case",
    [
        "input_ndim",
        "weight_ndim",
        "bias_ndim",
        "channel_mismatch",
        "groups_zero",
        "channels_not_divisible",
        "bias_size",
        "bad_stride",
        "bad_padding",
        "bad_output_padding",
        "bad_dilation",
        "bad_pair_length",
    ],
)
def test_accuracy_conv_transpose2d_invalid_inputs(case):
    dtype = torch.float32
    inp = torch.randn((1, 2, 4, 4), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((2, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((3,), dtype=dtype, device=flag_gems.device)
    kwargs = {
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "dilation": 1,
    }

    if case == "input_ndim":
        inp = inp.squeeze(0)
    elif case == "weight_ndim":
        weight = weight[0]
    elif case == "bias_ndim":
        bias = bias[None, :]
    elif case == "channel_mismatch":
        weight = torch.randn((3, 3, 3, 3), dtype=dtype, device=flag_gems.device)
    elif case == "groups_zero":
        kwargs["groups"] = 0
    elif case == "channels_not_divisible":
        kwargs["groups"] = 3
    elif case == "bias_size":
        bias = torch.randn((4,), dtype=dtype, device=flag_gems.device)
    elif case == "bad_stride":
        kwargs["stride"] = 0
    elif case == "bad_padding":
        kwargs["padding"] = -1
    elif case == "bad_output_padding":
        kwargs["stride"] = 2
        kwargs["output_padding"] = 2
    elif case == "bad_dilation":
        kwargs["dilation"] = 0
    elif case == "bad_pair_length":
        kwargs["stride"] = (1, 1, 1)

    with pytest.raises(AssertionError):
        with torch.no_grad():
            flag_gems.conv_transpose2d(inp, weight, bias=bias, **kwargs)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D[:2])
@pytest.mark.parametrize(
    "stride, padding, output_padding, dilation",
    [
        (1, 0, 0, 1),
        (2, 1, 0, 1),
        ((2, 2), (1, 1), (1, 1), 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("bias", [True, False])
def test_accuracy_conv_transpose2d_backward(
    shape, kernel, groups, stride, padding, output_padding, dilation, dtype, bias
):
    _disable_tf32()
    reduce_dim = (shape[1] // groups) * kernel[2] * kernel[3]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_weight = utils.to_reference(weight, True)

    if bias:
        bias_tensor = torch.randn(
            [kernel[1] * groups],
            dtype=dtype,
            device=flag_gems.device,
            requires_grad=True,
        )
        bias_ref = utils.to_reference(bias_tensor, True)
    else:
        bias_tensor = None
        bias_ref = None

    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=bias_tensor,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)
    ref_grad = utils.to_reference(out_grad, True)

    if bias:
        ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
        )
        res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
            res_out, (inp, weight, bias_tensor), out_grad
        )
    else:
        ref_in_grad, ref_weight_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        res_in_grad, res_weight_grad = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=kernel[2])
    utils.gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=shape[1]
    )
    if bias:
        utils.gems_assert_close(res_bias_grad, ref_bias_grad, dtype)
