import random

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference
from .conftest import QUICK_MODE


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize(
    "src_shape", [(32, 8, 4)] if QUICK_MODE else [(128, 16, 4), (256, 32, 8)]
)
@pytest.mark.parametrize(
    "inp_shape", [(64, 16, 8)] if QUICK_MODE else [(512, 128, 32), (1024, 64, 16)]
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("reduce", ["sum", "prod", "amax", "amin", "mean"])
def test_scatter_reduce_two_(src_shape, inp_shape, dim, dtype, reduce):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    size_dim = min(src_shape[dim], inp_shape[dim])

    index_shape = [
        random.randint(1, min(src_shape[0], inp_shape[0])),
        random.randint(1, min(src_shape[1], inp_shape[1])),
        random.randint(1, min(src_shape[2], inp_shape[2])),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    m, n, o = index_shape
    index_size_dim = index_shape[dim]
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = to_reference(inp.clone(), upcast=True)
    ref_index = to_reference(index)
    ref_src = to_reference(src, upcast=True)
    ref_out = ref_inp.scatter_reduce_(dim, ref_index, ref_src, reduce=reduce)
    with flag_gems.use_gems():
        res_out = inp.scatter_reduce_(dim, index, src, reduce=reduce)

    gems_assert_close(res_out, ref_out, dtype)
