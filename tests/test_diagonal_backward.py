import itertools
import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


def get_dim1_dim2(o_rank):
    dims = list(range(-o_rank, o_rank))
    return [
        p for p in itertools.permutations(dims, 2) if (p[0] % o_rank) != (p[1] % o_rank)
    ]


def get_shape_and_dims():
    shapes = utils.SPECIAL_SHAPES
    result = []

    for s in shapes:
        dim_pairs = get_dim1_dim2(len(s))
        if dim_pairs:
            dim1, dim2 = random.choice(dim_pairs)
            result.append((s, dim1, dim2))

    return result


@pytest.mark.diagonal_backward
@pytest.mark.skipif(flag_gems.device == "kunlunxin", reason="Temporary skip")
@pytest.mark.parametrize("shape, dim1, dim2", get_shape_and_dims())
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagonal_backward(shape, dtype, dim1, dim2, offset):
    if flag_gems.vendor_name == "mthreads":
        torch.manual_seed(123)
        torch.musa.manual_seed_all(123)

    torch.empty(1, device=flag_gems.device, requires_grad=True).backward()
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.diagonal(ref_inp, offset, dim1, dim2)
    with flag_gems.use_gems():
        res_out = torch.diagonal(inp, offset, dim1, dim2)

    out_grad = torch.randn_like(res_out.cpu()).to(device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(res_in_grad, ref_in_grad)
