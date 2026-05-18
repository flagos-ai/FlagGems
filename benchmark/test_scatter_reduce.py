import pytest
import torch

from flag_gems.utils import shape_utils

from . import base, consts


class TensorSelectBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # The generic comprehensive 2D shapes include very large tensors that
        # make scatter/scatter_reduce benchmarks time out.
        return []


class ScatterReduceBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_SHAPE_FILES = "core_shapes.yaml"

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return []


def gather_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)

    dim = -1
    size_dim = shape[dim]
    index_shape = list(shape)
    index_shape[dim] = 2 * shape[dim]
    index = torch.randint(0, size_dim, index_shape, dtype=torch.long, device=device)
    yield inp, dim, index


def scatter_input_fn_factory(reduce=None):
    def inner(shape, dtype, device):
        input_gen = gather_input_fn(shape, dtype, device)
        inp, dim, index = next(input_gen)
        src_shape = [size + 16 for size in index.shape]
        src = torch.randn(src_shape, dtype=dtype, device=device)

        if reduce is None:
            yield inp, dim, index, src
        else:
            yield inp, dim, index, src, reduce

    return inner


def scatter_inplace_input_fn_factory(reduce=None):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = -1
        size_dim = shape[dim]
        index = torch.randint(0, size_dim, shape, dtype=torch.long, device=device)
        src = torch.randn(shape, dtype=dtype, device=device)

        if reduce is None:
            yield inp, dim, index, src
        else:
            yield inp, dim, index, src, reduce

    return inner


def gather_scatter_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    data_shape = list(inp.shape)
    data_shape[dim] = index.shape[dim]
    data = torch.empty(data_shape, dtype=inp.dtype, device=inp.device)
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, data, data]])
    return io_amount * 1e-9 / (latency * 1e-3)


def scatter_reduce_two_input_fn_factory(reduce="sum", include_self=True):
    def inner(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = -1
        src_shape = list(shape)
        src_shape[dim] = max(1, shape[dim] // 2)
        src = torch.randn(src_shape, dtype=dtype, device=device)
        index = torch.arange(src_shape[dim], dtype=torch.long, device=device).expand(
            src_shape
        )
        yield inp, dim, index, src, {"reduce": reduce, "include_self": include_self}

    return inner


def scatter_reduce_two_out_input_fn_factory(reduce="sum", include_self=True):
    def inner(shape, dtype, device):
        for inp, dim, index, src, kwargs in scatter_reduce_two_input_fn_factory(
            reduce, include_self
        )(shape, dtype, device):
            yield inp, dim, index, src, kwargs, {"out": torch.empty_like(inp)}

    return inner


SCATTER_REDUCE_TWO_FORWARD_CASES = [
    ("scatter_reduce_two", "sum", True),
    ("scatter_reduce_two", "mean", False),
    ("scatter_reduce_two", "prod", True),
    ("scatter_reduce_two", "amax", False),
    ("scatter_reduce_two", "amin", True),
]

SCATTER_REDUCE_TWO_INPLACE_CASES = [
    ("scatter_reduce_two_", "sum", True),
    ("scatter_reduce_two_", "prod", True),
    ("scatter_reduce_two_", "amax", True),
    ("scatter_reduce_two_", "amin", True),
    ("scatter_reduce_two_", "mean", True),
]

SCATTER_REDUCE_TWO_OUT_CASES = [
    ("scatter_reduce_two_out", "sum", True),
    ("scatter_reduce_two_out", "mean", False),
    ("scatter_reduce_two_out", "prod", True),
    ("scatter_reduce_two_out", "amax", False),
    ("scatter_reduce_two_out", "amin", True),
]


@pytest.mark.scatter_reduce
def test_scatter_reduce_add():
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce",
        torch_op=torch.scatter,
        input_fn=scatter_input_fn_factory("add"),
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.scatter_reduce
def test_scatter_reduce_multiply():
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce",
        torch_op=torch.scatter,
        input_fn=scatter_input_fn_factory("multiply"),
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


@pytest.mark.scatter_reduce_
def test_scatter_reduce_add_inplace():
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_,
        input_fn=scatter_inplace_input_fn_factory("add"),
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_
def test_scatter_reduce_multiply_inplace():
    bench = TensorSelectBenchmark(
        op_name="scatter_reduce_",
        torch_op=torch.Tensor.scatter_,
        input_fn=scatter_inplace_input_fn_factory("multiply"),
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize(
    "op_name, reduce, include_self", SCATTER_REDUCE_TWO_FORWARD_CASES
)
def test_scatter_reduce_two(op_name, reduce, include_self):
    bench = ScatterReduceBenchmark(
        op_name=op_name,
        torch_op=torch.scatter_reduce,
        input_fn=scatter_reduce_two_input_fn_factory(reduce, include_self),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize(
    "op_name, reduce, include_self", SCATTER_REDUCE_TWO_INPLACE_CASES
)
def test_scatter_reduce_two_(op_name, reduce, include_self):
    bench = ScatterReduceBenchmark(
        op_name=op_name,
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=scatter_reduce_two_input_fn_factory(reduce, include_self),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce_two_out
@pytest.mark.parametrize("op_name, reduce, include_self", SCATTER_REDUCE_TWO_OUT_CASES)
def test_scatter_reduce_two_out(op_name, reduce, include_self):
    bench = ScatterReduceBenchmark(
        op_name=op_name,
        torch_op=torch.scatter_reduce,
        input_fn=scatter_reduce_two_out_input_fn_factory(reduce, include_self),
        get_gbps=gather_scatter_gbps,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
