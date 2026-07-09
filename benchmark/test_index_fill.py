import math

import pytest
import torch

from flag_gems.ops import (
    index_fill_scalar,
    index_fill_scalar_,
    index_fill_scalar_out,
    index_fill_tensor,
    index_fill_tensor_,
    index_fill_tensor_out,
)

from . import base, consts


INDEX_RATIOS = ("one", "1/16", "1/2", "full")
INDEX_FILL_DTYPES = [torch.float16, torch.float32, torch.bfloat16]


class IndexFillBenchmark(base.GenericBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS + ["gbps"]
    DEFAULT_SHAPES = [
        (1024,),
        (4096, 256),
        (4096, 4096),
    ]
    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        self.shapes = list(self.DEFAULT_SHAPES)
        if (
            base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            and not base.Config.query
        ):
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))

    def set_more_shapes(self):
        return [
            (8192, 4096),
            (64, 1024, 128),
            (200, 40999, 3),
        ]


def _generate_input(shape, dtype, device):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randint(-10, 10, shape, dtype=dtype, device=device)


def _dim_for_shape(shape):
    return 0 if len(shape) == 1 else 1


def _index_len(dim_size, ratio):
    if ratio == "one":
        return 1
    if ratio == "1/16":
        return max(1, dim_size // 16)
    if ratio == "1/2":
        return max(1, dim_size // 2)
    if ratio == "full":
        return dim_size
    raise ValueError(f"Unknown index ratio: {ratio}")


def _make_index(dim_size, index_len, device):
    return torch.randperm(dim_size, device=device)[:index_len]


def _scalar_value(dtype):
    if dtype == torch.bool:
        return True
    if dtype.is_floating_point:
        return 3.14159
    return 3


def _base_inputs(shape, dtype, device):
    dim = _dim_for_shape(shape)
    dim_size = shape[dim]
    seen_index_lens = set()
    for ratio in INDEX_RATIOS:
        index_len = _index_len(dim_size, ratio)
        if index_len in seen_index_lens:
            continue
        seen_index_lens.add(index_len)
        inp = _generate_input(shape, dtype, device)
        index = _make_index(dim_size, index_len, device)
        yield inp, dim, index


def _selected_numel(inp, dim, index):
    return math.prod(inp.shape) // inp.size(dim) * index.numel()


def _inplace_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    bytes_per_elem = inp.element_size()
    io_amount = index.numel() * index.element_size()
    io_amount += _selected_numel(inp, dim, index) * bytes_per_elem
    return io_amount * 1e-9 / (latency * 1e-3)


def _out_of_place_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    bytes_per_elem = inp.element_size()
    io_amount = inp.numel() * bytes_per_elem * 2
    io_amount += index.numel() * index.element_size()
    io_amount += _selected_numel(inp, dim, index) * bytes_per_elem
    return io_amount * 1e-9 / (latency * 1e-3)


def index_fill_scalar_input_fn(shape, dtype, device):
    for inp, dim, index in _base_inputs(shape, dtype, device):
        yield inp, dim, index, _scalar_value(dtype)


def index_fill_tensor_input_fn(shape, dtype, device):
    for inp, dim, index in _base_inputs(shape, dtype, device):
        value = torch.tensor(_scalar_value(dtype), dtype=dtype, device=device)
        yield inp, dim, index, value


def index_fill_scalar_out_input_fn(shape, dtype, device):
    for inp, dim, index in _base_inputs(shape, dtype, device):
        out = torch.empty_like(inp)
        yield inp, dim, index, _scalar_value(dtype), {"out": out}


def index_fill_tensor_out_input_fn(shape, dtype, device):
    for inp, dim, index in _base_inputs(shape, dtype, device):
        value = torch.tensor(_scalar_value(dtype), dtype=dtype, device=device)
        out = torch.empty_like(inp)
        yield inp, dim, index, value, {"out": out}


@pytest.mark.index_fill_scalar
def test_index_fill_scalar():
    bench = IndexFillBenchmark(
        op_name="index_fill_scalar",
        input_fn=index_fill_scalar_input_fn,
        torch_op=torch.index_fill,
        gems_op=index_fill_scalar,
        dtypes=INDEX_FILL_DTYPES,
        get_gbps=_out_of_place_gbps,
    )
    bench.run()


@pytest.mark.index_fill_scalar_
def test_index_fill_scalar_():
    bench = IndexFillBenchmark(
        op_name="index_fill_scalar_",
        input_fn=index_fill_scalar_input_fn,
        torch_op=torch.Tensor.index_fill_,
        gems_op=index_fill_scalar_,
        dtypes=INDEX_FILL_DTYPES,
        is_inplace=True,
        get_gbps=_inplace_gbps,
    )
    bench.run()


@pytest.mark.index_fill_tensor
def test_index_fill_tensor():
    bench = IndexFillBenchmark(
        op_name="index_fill_tensor",
        input_fn=index_fill_tensor_input_fn,
        torch_op=torch.index_fill,
        gems_op=index_fill_tensor,
        dtypes=INDEX_FILL_DTYPES,
        get_gbps=_out_of_place_gbps,
    )
    bench.run()


@pytest.mark.index_fill_tensor_
def test_index_fill_tensor_():
    bench = IndexFillBenchmark(
        op_name="index_fill_tensor_",
        input_fn=index_fill_tensor_input_fn,
        torch_op=torch.Tensor.index_fill_,
        gems_op=index_fill_tensor_,
        dtypes=INDEX_FILL_DTYPES,
        is_inplace=True,
        get_gbps=_inplace_gbps,
    )
    bench.run()


@pytest.mark.index_fill_scalar_out
def test_index_fill_scalar_out():
    bench = IndexFillBenchmark(
        op_name="index_fill_scalar_out",
        input_fn=index_fill_scalar_out_input_fn,
        torch_op=torch.ops.aten.index_fill.int_Scalar_out,
        gems_op=index_fill_scalar_out,
        dtypes=INDEX_FILL_DTYPES,
        is_inplace=True,
        get_gbps=_out_of_place_gbps,
    )
    bench.run()


@pytest.mark.index_fill_tensor_out
def test_index_fill_tensor_out():
    bench = IndexFillBenchmark(
        op_name="index_fill_tensor_out",
        input_fn=index_fill_tensor_out_input_fn,
        torch_op=torch.ops.aten.index_fill.int_Tensor_out,
        gems_op=index_fill_tensor_out,
        dtypes=INDEX_FILL_DTYPES,
        is_inplace=True,
        get_gbps=_out_of_place_gbps,
    )
    bench.run()
