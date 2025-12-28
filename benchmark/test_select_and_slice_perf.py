import random

import numpy as np
import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import (
    GenericBenchmark,
    GenericBenchmark2DOnly,
    generate_tensor_input,
    vendor_name,
)
from flag_gems.utils import shape_utils


class TensorSelectBenchmark(GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        if (
            vendor_name == "kunlunxin"
        ):  # Speed Up Benchmark Test, Big Shape Will Cause Timeout
            return []
        shapes = super().set_more_shapes()
        shapes = [
            # this filter is for scatter
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes


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


def mask_select_gbps(bench_fn_args, latency):
    mask = bench_fn_args[1]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [mask]])
    io_amount += 2 * int(torch.sum(mask))
    return io_amount * 1e-9 / (latency * 1e-3)


def index_select_gbps(bench_fn_args, latency):
    inp = bench_fn_args[0]
    dim = bench_fn_args[1]
    io_amount = shape_utils.size_in_bytes(inp) * 2 // inp.size(dim)
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.index_select
@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, gbps_fn, dtypes",
    [
        pytest.param(
            "index_select",
            torch.index_select,
            index_select_input_fn,
            index_select_gbps,
            FLOAT_DTYPES,
            marks=pytest.mark.index_select,
        ),
    ],
)
def test_perf_index_select(op_name, torch_op, input_fn, gbps_fn, dtypes):
    bench = TensorSelectBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        get_gbps=gbps_fn,
    )
    bench.run()


@pytest.mark.masked_select
@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, gbps_fn, dtypes",
    [
        pytest.param(
            "masked_select",
            torch.masked_select,
            masked_select_input_fn,
            mask_select_gbps,
            FLOAT_DTYPES,
            marks=pytest.mark.masked_select,
        ),
    ],
)
def test_perf_masked_select(op_name, torch_op, input_fn, gbps_fn, dtypes):
    bench = TensorSelectBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        get_gbps=gbps_fn,
    )
    bench.run()


def gather_scatter_gbps(bench_fn_args, latency):
    inp, dim, index = bench_fn_args[:3]
    data_shape = list(inp.shape)
    data_shape[dim] = index.shape[dim]
    data = torch.empty(data_shape, dtype=inp.dtype, device=inp.device)
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, data, data]])
    return io_amount * 1e-9 / (latency * 1e-3)


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


@pytest.mark.scatter
def test_perf_scatter():
    bench = TensorSelectBenchmark(
        op_name="scatter.src",
        torch_op=torch.scatter,
        input_fn=scatter_input_fn_factory(),
        get_gbps=gather_scatter_gbps,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.scatter
def test_perf_scatter_add():
    bench = TensorSelectBenchmark(
        op_name="scatter.reduce",
        torch_op=torch.scatter,
        input_fn=scatter_input_fn_factory("add"),
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.scatter_add_
def test_perf_scatter_add_():
    def scatter_input_fn(shape, dtype, device):
        input_gen = gather_input_fn(shape, dtype, device)
        inp, dim, index = next(input_gen)
        src_shape = list(size + 16 for size in index.shape)
        src = torch.randn(src_shape, dtype=dtype, device=device)

        yield inp, dim, index, src

    bench = TensorSelectBenchmark(
        op_name="scatter_add_",
        torch_op=torch.Tensor.scatter_add_,
        input_fn=scatter_input_fn,
        get_gbps=gather_scatter_gbps,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.scatter_multiply
@pytest.mark.scatter
def test_perf_scatter_multiply():
    bench = TensorSelectBenchmark(
        op_name="scatter.reduce",
        torch_op=torch.scatter,
        input_fn=scatter_input_fn_factory("multiply"),
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


@pytest.mark.scatter_
def test_perf_scatter_inplace():
    bench = TensorSelectBenchmark(
        op_name="scatter_.src",
        torch_op=torch.Tensor.scatter_,
        input_fn=scatter_inplace_input_fn_factory(),
        get_gbps=gather_scatter_gbps,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_
def test_perf_scatter_add_inplace():
    bench = TensorSelectBenchmark(
        op_name="scatter_.reduce",
        torch_op=torch.Tensor.scatter_,
        input_fn=scatter_inplace_input_fn_factory("add"),
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_
def test_perf_scatter_multiply_inplace():
    bench = TensorSelectBenchmark(
        op_name="scatter_.reduce",
        torch_op=torch.Tensor.scatter_,
        input_fn=scatter_inplace_input_fn_factory("multiply"),
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float16, torch.float32],
        is_inplace=True,
    )
    bench.run()


def gather_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)

    dim = -1
    size_dim = shape[dim]
    index_shape = list(shape)
    index_shape[dim] = 2 * shape[dim]
    index = torch.randint(0, size_dim, index_shape, dtype=torch.long, device=device)
    yield inp, dim, index


@pytest.mark.gather
def test_perf_gather():
    bench = TensorSelectBenchmark(
        op_name="gather",
        torch_op=torch.gather,
        input_fn=gather_input_fn,
        get_gbps=gather_scatter_gbps,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


def slice_scatter_gbps(bench_fn_args, latency):
    inp = bench_fn_args[0]
    src = bench_fn_args[1]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [inp, src, src]])
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.skipif(vendor_name == "kunlunxin", reason="RESULT TODOFIX")
@pytest.mark.gather
def test_perf_gather_backward():
    bench = TensorSelectBenchmark(
        op_name="gather",
        torch_op=torch.gather,
        input_fn=gather_input_fn,
        get_gbps=gather_scatter_gbps,
        dtypes=[torch.float32],
        is_backward=True,
    )
    bench.run()


@pytest.mark.slice_scatter
def test_slice_scatter_perf():
    def slice_scatter_input_fn(shape, dtype, device):
        dim = 0 if len(shape) == 1 else 1
        start = 0
        end = shape[dim]
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
        get_gbps=slice_scatter_gbps,
    )
    bench.run()


@pytest.mark.select_scatter
def test_select_scatter_perf():
    def select_scatter_input_fn(shape, dtype, device):
        dim = 0 if len(shape) == 1 else 1
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
        get_gbps=slice_scatter_gbps,
    )
    bench.run()


def index_add_gbps(bench_fn_args, latency):
    index = bench_fn_args[2]
    src = bench_fn_args[3]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, src, src]])
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.index_add
def test_index_add_perf():
    def index_add_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = 0 if len(shape) == 1 else 1
        src_shape = list(inp.shape)
        index_max = src_shape[dim]
        index_len = index_max // 2 if index_max >= 2 else 1
        index = torch.randperm(index_len, device=device)
        src_shape[dim] = index_len
        src = torch.randn(src_shape, dtype=dtype, device=device)
        yield inp, dim, index, src

    bench = TensorSelectBenchmark(
        op_name="index_add",
        torch_op=torch.index_add,
        input_fn=index_add_input_fn,
        dtypes=[torch.float16, torch.float32],
        get_gbps=index_add_gbps,
    )
    bench.run()


@pytest.mark.index_add_
def test_index_add__perf():
    def index_add__input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        dim = 0 if len(shape) == 1 else 1
        src_shape = list(inp.shape)
        index_max = src_shape[dim]
        index_len = index_max // 2 if index_max >= 2 else 1
        index = torch.randperm(index_len, device=device)
        src_shape[dim] = index_len
        src = torch.randn(src_shape, dtype=dtype, device=device)
        yield inp, dim, index, src

    bench = TensorSelectBenchmark(
        op_name="index_add_",
        torch_op=torch.Tensor.index_add_,
        input_fn=index_add__input_fn,
        dtypes=[torch.float16, torch.float32],
        get_gbps=index_add_gbps,
    )
    bench.run()


def gen_indices(input_shape, indices_shape, accumulate):
    indices = []
    for i, shape in enumerate(indices_shape):
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))
    return indices


def gen_indices_bool(input_shape, indices_shape, accumulate, is_bool):
    indices = []

    if is_bool:
        mask_shape = indices_shape[0]

        mask = torch.randint(
            0, 2, size=mask_shape, dtype=torch.bool, device=flag_gems.device
        )
        return [mask]

    else:
        for i, shape in enumerate(indices_shape):
            index = np.random.choice(
                np.arange(input_shape[i]), size=shape, replace=accumulate
            )
            indices.append(torch.tensor(index, device=flag_gems.device))
        return indices


def index_put_input_fn(accumulate):
    def inner(shapes, dtype, device):
        input_shape, indices_shape, values_shape, is_bool = shapes
        inp = torch.randn(
            input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )

        indices = gen_indices_bool(input_shape, indices_shape, accumulate, is_bool)

        if is_bool:
            K = indices[0].sum().item()
            values = torch.randn(
                (K,), dtype=dtype, device=flag_gems.device, requires_grad=False
            )
        else:
            values = torch.randn(
                values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
            )
        yield inp, indices, values, accumulate

    return inner


class IndexPutAccFalseBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        INDEX_PUT_SHAPE = (
            ((2**28,), ((2**16,),), (2**16,), False),
            ((32, 32), ((8,), (8,)), (8,), False),
            ((32, 32), ((8,), (2, 8)), (8,), False),
            ((32, 32), ((2, 8),), (32,), False),
            ((1024, 1024), ((64,), (64,)), (64,), False),
            (
                (1024, 1024),
                (
                    (64,),
                    (
                        4,
                        64,
                    ),
                ),
                (64,),
                False,
            ),
            (
                (1024, 1024),
                (
                    (
                        4,
                        64,
                    ),
                ),
                (1024,),
                False,
            ),
            ((512, 512, 512), ((128,), (128,), (128,)), (128,), False),
            ((512, 512, 512), ((2, 128), (128,), (128,)), (128,), False),
            ((512, 512, 512), ((2, 128),), (512,), False),
            ((100,), ((100,),), (100,), True),
            ((32, 32), ((32, 32),), (32, 32), True),
            ((16, 16, 4), ((16, 16, 4),), (16, 16, 4), True),
            ((1024, 1024), ((1024, 1024),), (1024 * 1024,), True),
        )
        self.shapes = INDEX_PUT_SHAPE
        return None


@pytest.mark.index_put
def test_index_put_acc_false_perf():
    bench = IndexPutAccFalseBenchmark(
        op_name="index_put",
        torch_op=torch.index_put,
        input_fn=index_put_input_fn(False),
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.index_put_
def test_index_put__acc_false_perf():
    bench = IndexPutAccFalseBenchmark(
        op_name="index_put_",
        torch_op=torch.index_put_,
        input_fn=index_put_input_fn(False),
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


class IndexPutAccTrueBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        INDEX_PUT_SHAPE = (
            ((2**28,), ((2**16,),), (2**16,), False),
            ((32, 32), ((8,), (8,)), (8,), False),
            ((1024, 1024), ((64,), (64,)), (64,), False),
            ((512, 512, 512), ((128,), (128,), (128,)), (128,), False),
            ((512, 512, 512), ((2, 128), (2, 128), (2, 128)), (2, 128), False),
            ((512, 512), ((512, 512),), (512 * 512,), True),
            ((64, 64, 64), ((64, 64, 64),), (64**3,), True),
        )
        self.shapes = INDEX_PUT_SHAPE
        return None


@pytest.mark.index_put
def test_index_put_acc_true_perf():
    bench = IndexPutAccTrueBenchmark(
        op_name="index_put",
        torch_op=torch.index_put,
        input_fn=index_put_input_fn(True),
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


@pytest.mark.index_put_
def test_index_put__acc_true_perf():
    bench = IndexPutAccTrueBenchmark(
        op_name="index_put_",
        torch_op=torch.index_put_,
        input_fn=index_put_input_fn(True),
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()


class IndexAccBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        INDEX_SHAPE = (
            ((2**28,), ((2**16,),)),
            ((32, 32), ((8,), (8,))),
            ((32, 32), ((8,), (2, 8))),
            ((32, 32), ((2, 8),)),
            ((1024, 1024), ((64,), (64,))),
            ((512, 512, 512), ((128,), (128,), (128,))),
            ((512, 512, 512), ((2, 128), (2, 128), (2, 128))),
            ((512, 512, 512), ((2, 128), (128,), (128,))),
            ((512, 512, 512), ((2, 128),)),
            (
                (64, 64, 64),
                (
                    (2, 8),
                    (2, 8),
                ),
            ),
        )
        self.shapes = INDEX_SHAPE
        return None


def index_input_fn(shapes, dtype, device):
    input_shape, indices_shape = shapes
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    indices = gen_indices(input_shape, indices_shape, True)
    yield inp, indices


@pytest.mark.index
def test_index_acc_perf():
    gems_op = flag_gems.index
    bench = IndexAccBenchmark(
        op_name="index",
        torch_op=torch.ops.aten.index,
        input_fn=index_input_fn,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.set_gems(gems_op)
    bench.run()
