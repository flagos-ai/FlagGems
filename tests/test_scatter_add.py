import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    # QUICK_MODE: single dtype/shape for fast smoke test
    FLOAT_DTYPES = [torch.float32]
    # small 3D src for quick smoke
    SOURCE_SHAPES = [(32, 8, 4)]
    # small 3D inp for quick smoke
    INPUT_SHAPES = [(64, 16, 8)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    # 3D shapes covering small/medium src and large inp for scatter_add coverage
    SOURCE_SHAPES = [(128, 16, 4), (256, 32, 8)]
    # large inp to test scatter accumulation along all dims
    INPUT_SHAPES = [(512, 128, 32), (1024, 64, 16)]


@pytest.mark.scatter_add
@pytest.mark.parametrize("src_shape", SOURCE_SHAPES)
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_scatter_add(src_shape, inp_shape, dim, dtype):
    utils.init_seed(0)

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
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_add(ref_inp, dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = torch.scatter_add(inp, dim, index, src)

    utils.gems_assert_close(res_out, ref_out, dtype)
