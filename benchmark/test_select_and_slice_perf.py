import random

import pytest
import torch

from .attri_util import FLOAT_DTYPES
from .performance_utils import GenericBenchmark2DOnly, generate_tensor_input


class TensorSelectBenchmark(GenericBenchmark2DOnly):
    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        return [
            # this filter is for scatter
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]


def index_select_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    threshold = 0.1
    dim = 0
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(0, index_size, [floor(index_size * threshold)], device=device)
    yield inp, dim, index


def masked_select_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    mask = generate_tensor_input(shape, cur_dtype, device) < 0.3
    yield inp, mask


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "index_select",
            torch.index_select,
            index_select_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.index_select,
        ),
        pytest.param(
            "masked_select",
            torch.masked_select,
            masked_select_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.masked_select,
        ),
    ],
)
def test_generic_reduction_benchmark(op_name, torch_op, input_fn, dtypes):
    bench = TensorSelectBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()


@pytest.mark.scatter
def test_perf_scatter():
    def scatter_input_fn(shape, dtype, device):
        batch, size = shape
        src_shape = [batch // 16, size // 16]
        inp = torch.randn(shape, dtype=dtype, device=device)
        src = torch.randn(src_shape, dtype=dtype, device=device)

        dim = random.choice([0, 1])
        size_dim = min(src_shape[dim], shape[dim])

        index_shape = [
            random.randint(1, min(src_shape[0], shape[0])),
            random.randint(1, min(src_shape[1], shape[1])),
        ]
        index = torch.empty(tuple(index_shape), dtype=torch.long, device=device)

        m, n = index_shape

        index_size_dim = index_shape[dim]
        # make unique indices
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                ii = [i, j]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

        yield inp, dim, index, src

    bench = TensorSelectBenchmark(
        op_name="scatter",
        torch_op=torch.scatter,
        input_fn=scatter_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.gather
def test_perf_gather():
    def gather_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)

        dim = random.choice([0, 1])
        size_dim = shape[dim]
        index_shape = [
            random.randint(1, shape[0]),
            random.randint(1, shape[1]),
        ]
        index = torch.empty(tuple(index_shape), dtype=torch.long, device=device)

        m, n = index_shape

        index_size_dim = index_shape[dim]
        # make unique indices
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                ii = [i, j]
                ii[dim] = slice(0, index.size(dim) + 1)
                index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

        yield inp, dim, index

    bench = TensorSelectBenchmark(
        op_name="gather",
        torch_op=torch.gather,
        input_fn=gather_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.slice_scatter
def test_slice_scatter_perf():
    def slice_scatter_input_fn(shape, dtype, device):
        dim = random.choice([0, 1])
        start = 16
        end = 1024
        step = 2

        inp = torch.randn(shape, dtype=dtype, device=device)

        range = end - start
        valid_shape = list(inp.shape)
        if end < start:
            range = 0
        elif (end - start) > valid_shape[dim]:
            range = valid_shape[dim]
            start = 0
            end = valid_shape[dim]

        valid_shape[dim] = (range + (step - 1)) // step
        src = torch.randn(valid_shape, dtype=dtype, device=device)
        yield inp, src, dim, start, end, step

    bench = TensorSelectBenchmark(
        op_name="slice_scatter",
        torch_op=torch.slice_scatter,
        input_fn=slice_scatter_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.select_scatter
def test_select_scatter_perf():
    def select_scatter_input_fn(shape, dtype, device):
        dim = random.choice([0, 1])
        index = random.randint(0, shape[dim] - 1)
        inp = torch.randn(shape, dtype=dtype, device=device)

        src_shape = list(inp.shape)
        del src_shape[dim]
        src = torch.randn(src_shape, dtype=dtype, device=device)

        yield inp, src, dim, index

    bench = TensorSelectBenchmark(
        op_name="select_scatter",
        torch_op=torch.select_scatter,
        input_fn=select_scatter_input_fn,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
