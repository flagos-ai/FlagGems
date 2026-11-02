import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.clone
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clone(ref_inp)
    res_out = flag_gems.clone(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.clone
@pytest.mark.parametrize(
    "case",
    [
        "contiguous",
        "expanded",
        "stepped_slice",
        "transposed",
        "channels_last",
        "permuted",
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone_preserve_format_layout(dtype, case):
    torch.manual_seed(0)
    if case == "channels_last":
        inp = torch.randn((2, 3, 4, 5), dtype=dtype, device=flag_gems.device).to(
            memory_format=torch.channels_last
        )
    elif case == "permuted":
        inp = torch.randn((2, 3, 4), dtype=dtype, device=flag_gems.device).permute(
            2, 0, 1
        )
    else:
        base = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
        if case == "contiguous":
            inp = base
        elif case == "expanded":
            inp = base[:, :1].expand(4, 8)
        elif case == "stepped_slice":
            inp = base[:, ::2]
        else:
            inp = base.t()

    ref_out = torch.clone(utils.to_reference(inp))
    res_out = flag_gems.clone(inp)

    assert res_out.shape == ref_out.shape
    assert tuple(res_out.stride()) == tuple(ref_out.stride())
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.clone
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone_non_contiguous_values(dtype):
    # Cloning a non-contiguous input must produce a tensor equal to the
    # logical values of the source.
    torch.manual_seed(0)
    base = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    for inp in (base[:, ::2], base.t()):
        ref_out = torch.clone(utils.to_reference(inp))
        res_out = flag_gems.clone(inp)
        utils.gems_assert_equal(res_out, ref_out.to(res_out.dtype))


@pytest.mark.clone
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
def test_clone_autograd(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)

    res_out = flag_gems.clone(inp)
    ref_out = torch.clone(ref_inp)

    (res_in_grad,) = torch.autograd.grad(res_out.sum(), [inp])
    (ref_in_grad,) = torch.autograd.grad(ref_out.sum(), [ref_inp])

    utils.gems_assert_close(res_in_grad, ref_in_grad.to(dtype), dtype)


@pytest.mark.clone
def test_clone_autograd_non_contiguous():
    torch.manual_seed(0)
    base = torch.randn((4, 8), device=flag_gems.device, requires_grad=True)
    inp = base[:, ::2]
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)

    res_out = flag_gems.clone(inp)
    ref_out = torch.clone(ref_inp)

    (res_in_grad,) = torch.autograd.grad(res_out.sum(), [inp])
    (ref_in_grad,) = torch.autograd.grad(ref_out.sum(), [ref_inp])

    utils.gems_assert_close(res_in_grad, ref_in_grad.to(torch.float32), torch.float32)
