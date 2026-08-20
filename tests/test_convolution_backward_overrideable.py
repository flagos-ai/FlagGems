import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Restricted dtype list: the generator limits the tested precisions for numerical-stability reasons (see worktree).
FLOAT_DTYPES = [torch.float16, torch.float32]


PADDINGS = [0, 1]


# shape = (input_shape, weight_shape, groups)
SHAPE_CONV = [
    ((1, 2, 5, 5), (1, 2, 3, 3), 1),
    ((2, 3, 9, 9), (4, 3, 3, 3), 1),
    ((32, 8, 8, 8), (32, 8, 2, 2), 1),
    ((2, 4, 7, 7), (6, 2, 3, 3), 2),
]


STRIDES = [1, 2]


def _make_inputs(shape, kernel, groups, stride, padding, dtype):
    torch.backends.cudnn.allow_tf32 = False
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    # Forward to determine grad_output shape.
    ref_out = torch.nn.functional.conv2d(
        ref_inp, ref_weight, bias=None, stride=stride, padding=padding, groups=groups
    )
    grad_out = torch.randn(ref_out.shape, dtype=dtype, device=flag_gems.device)
    ref_grad_out = utils.to_reference(grad_out, True)
    return inp, weight, grad_out, ref_inp, ref_weight, ref_grad_out


@pytest.mark.convolution_backward_overrideable
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_convolution_backward_overrideable(
    shape, kernel, groups, stride, padding, dtype
):
    inp, weight, grad_out, ref_inp, ref_weight, ref_grad_out = _make_inputs(
        shape, kernel, groups, stride, padding, dtype
    )

    stride_t = [stride, stride]
    padding_t = [padding, padding]
    dilation_t = [1, 1]
    output_padding = [0, 0]
    output_mask = [True, True, True]

    # Reference via the well-defined aten::convolution_backward.
    ref_gi, ref_gw, ref_gb = torch.ops.aten.convolution_backward(
        ref_grad_out,
        ref_inp,
        ref_weight,
        [weight.shape[0]],
        stride_t,
        padding_t,
        dilation_t,
        False,
        output_padding,
        groups,
        output_mask,
    )

    with flag_gems.use_gems():
        res_gi, res_gw, res_gb = torch.ops.aten.convolution_backward_overrideable(
            grad_out,
            inp,
            weight,
            stride_t,
            padding_t,
            dilation_t,
            False,
            output_padding,
            groups,
            output_mask,
        )

    utils.gems_assert_close(res_gi, ref_gi, dtype, reduce_dim=weight.shape[2])
    utils.gems_assert_close(res_gw, ref_gw, dtype, reduce_dim=weight.shape[0])
    utils.gems_assert_close(res_gb, ref_gb, dtype, reduce_dim=grad_out.shape[0])


@pytest.mark.convolution_backward_overrideable_out
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV)
@pytest.mark.parametrize("stride", STRIDES)
@pytest.mark.parametrize("padding", PADDINGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_convolution_backward_overrideable_out(
    shape, kernel, groups, stride, padding, dtype
):
    inp, weight, grad_out, ref_inp, ref_weight, ref_grad_out = _make_inputs(
        shape, kernel, groups, stride, padding, dtype
    )

    stride_t = [stride, stride]
    padding_t = [padding, padding]
    dilation_t = [1, 1]
    output_padding = [0, 0]
    output_mask = [True, True, True]

    ref_gi, ref_gw, ref_gb = torch.ops.aten.convolution_backward(
        ref_grad_out,
        ref_inp,
        ref_weight,
        [weight.shape[0]],
        stride_t,
        padding_t,
        dilation_t,
        False,
        output_padding,
        groups,
        output_mask,
    )

    out0 = torch.empty_like(inp)
    out1 = torch.empty_like(weight)
    out2 = torch.empty(weight.shape[0], dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res_gi, res_gw, res_gb = torch.ops.aten.convolution_backward_overrideable.out(
            grad_out,
            inp,
            weight,
            stride_t,
            padding_t,
            dilation_t,
            False,
            output_padding,
            groups,
            output_mask,
            out0=out0,
            out1=out1,
            out2=out2,
        )

    utils.gems_assert_close(res_gi, ref_gi, dtype, reduce_dim=weight.shape[2])
    utils.gems_assert_close(res_gw, ref_gw, dtype, reduce_dim=weight.shape[0])
    utils.gems_assert_close(res_gb, ref_gb, dtype, reduce_dim=grad_out.shape[0])
