import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.select_backward
@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (4, 8),
        (4, 8, 16),
        (2, 3, 4, 5),
        (8, 16, 32),
        (3, 7, 11),
        (2, 1, 4),
        (64, 512),
        (32, 256, 256),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_select_backward(shape, dtype, dim):
    ndim = len(shape)
    actual_dim = dim + ndim if dim < 0 else dim

    if actual_dim >= ndim:
        # Invalid input: dim out of range for shape
        return

    dim_size = shape[actual_dim]

    indices_to_test = [0, dim_size // 2]
    if dim_size > 1:
        indices_to_test.append(dim_size - 1)

    for index in indices_to_test:
        grad_shape = list(shape)
        grad_shape.pop(actual_dim)

        res_grad = torch.randn(
            grad_shape,
            dtype=dtype,
            device=flag_gems.device,
        )
        ref_grad = utils.to_reference(res_grad)

        ref_out = torch.ops.aten.select_backward(
            ref_grad,
            shape,
            actual_dim,
            index,
        )

        with flag_gems.use_gems():
            res_out = torch.ops.aten.select_backward(
                res_grad,
                shape,
                actual_dim,
                index,
            )

        assert res_out.shape == tuple(shape)
        assert res_out.dtype == res_grad.dtype

        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.select_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_select_backward_non_contiguous(dtype):
    base_shape = (8, 16, 32)

    res_x = torch.randn(base_shape, dtype=dtype, device=flag_gems.device)
    res_x = res_x.transpose(0, 1)  # non-contiguous

    shape = res_x.shape
    dim = 1
    index = min(5, shape[dim] - 1)

    grad_shape = list(shape)
    grad_shape.pop(dim)

    res_grad = torch.randn(grad_shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(res_grad)

    ref_out = torch.ops.aten.select_backward(ref_grad, shape, dim, index)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.select_backward(res_grad, shape, dim, index)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.select_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_select_backward_small_and_edge(dtype):
    shape = (1, 1, 1)
    dim = 0
    index = 0

    res_grad = torch.randn((1, 1), dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(res_grad)

    ref_out = torch.ops.aten.select_backward(ref_grad, shape, dim, index)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.select_backward(res_grad, shape, dim, index)

    utils.gems_assert_close(res_out, ref_out, dtype)
