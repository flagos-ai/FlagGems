import pytest
import torch

from . import base, consts


def index_copy_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = 0 if len(shape) == 1 else 1
    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max // 2 if index_max >= 2 else 1
    index = torch.randperm(index_len, device=device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=device)
    yield inp, dim, index, src


@pytest.mark.index_copy
def test_index_copy():
    bench = base.GenericBenchmark2DOnly(
        op_name="index_copy",
        torch_op=torch.index_copy,
        input_fn=index_copy_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.index_copy_
def test_index_copy_():
    bench = base.GenericBenchmark2DOnly(
        op_name="index_copy_",
        torch_op=torch.Tensor.index_copy_,
        input_fn=index_copy_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()
