import pytest
import torch

from . import base, consts

# index_reduce benchmark
INDEX_REDUCE_SHAPES = [
    (8, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]


class IndexReduceBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = INDEX_REDUCE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from index_reduce_input_fn(shape, cur_dtype, self.device)


def index_reduce_input_fn(shape, dtype, device):
    # Generate input tensors for index_reduce
    # shape: (M, N) - M is the size along reduction dimension
    # We'll reduce along dim=0
    inp = torch.full(shape, 2.0, dtype=dtype, device=device)
    source = torch.randn(shape[0], shape[1], dtype=dtype, device=device)
    index = torch.randint(0, shape[0], (shape[0],), dtype=torch.long, device=device)
    # Only pass tensor arguments, other params are fixed
    yield (inp, index, source)


def index_reduce_torch_op(inp, index, source):
    # Wrapper to call torch.index_reduce with fixed params
    return torch.index_reduce(inp, 0, index, source, "prod")


def index_reduce__torch_op(inp, index, source):
    # Wrapper to call in-place torch.index_reduce_ with fixed params
    return inp.clone().index_reduce_(0, index, source, "prod")


@pytest.mark.index_reduce
def test_index_reduce():
    bench = IndexReduceBenchmark(
        op_name="index_reduce",
        torch_op=index_reduce_torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.index_reduce_
def test_index_reduce_():
    bench = IndexReduceBenchmark(
        op_name="index_reduce_",
        torch_op=index_reduce__torch_op,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
