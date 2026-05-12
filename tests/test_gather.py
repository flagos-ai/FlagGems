import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INPUT_SHAPES = [(32, 8, 4)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    INPUT_SHAPES = [(512, 128, 32), (1024, 64, 16), (128, 32, 256)]

random.seed(time.time() // 100)


@pytest.mark.gather
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_gather(inp_shape, dim, dtype):
    # Test ndim mismatch raises IndexError (only once to avoid redundant checks)
    if dim == 0 and dtype == torch.float32:
        mismatch_inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
        mismatch_index = torch.zeros(
            inp_shape[:2], dtype=torch.long, device=flag_gems.device
        )
        with pytest.raises(IndexError):
            with flag_gems.use_gems():
                torch.gather(mismatch_inp, 0, mismatch_index)

    inp = torch.randn(
        inp_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    size_dim = inp_shape[dim]

    index_shape = [
        random.randint(1, inp_shape[0]),
        random.randint(1, inp_shape[1]),
        random.randint(1, inp_shape[2]),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_out = torch.gather(ref_inp, dim, ref_index)

    with flag_gems.use_gems():
        res_out = torch.gather(inp, dim, index)

    utils.gems_assert_equal(res_out, ref_out)

    if dtype in (torch.bfloat16,):
        return

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad)

    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)

    res_in_grad = utils.to_reference(res_in_grad)
    utils.gems_assert_equal(res_in_grad, ref_in_grad)
