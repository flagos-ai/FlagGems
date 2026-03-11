import pytest
import torch

from flag_gems.utils import shape_utils

from . import base


class ScatterReduceBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # Keep this benchmark on its dedicated shape list even in comprehensive mode.
        return []


def scatter_reduce_gbps(bench_fn_args, latency):
    inp, dim, index, src = bench_fn_args[:4]
    data_shape = list(inp.shape)
    data_shape[dim] = index.shape[dim]
    data = torch.empty(data_shape, dtype=inp.dtype, device=inp.device)
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, data, data]])
    return io_amount * 1e-9 / (latency * 1e-3)


def scatter_reduce_input_fn_factory(reduce="sum", include_self=True):
    def inner(shape, dtype, device):
        if dtype in (torch.int16, torch.int32, torch.int64):
            inp = torch.randint(-8, 8, shape, device=device).to(dtype)
        else:
            inp = torch.randn(shape, dtype=dtype, device=device)

        dim = -1
        src_shape = list(shape)
        src_shape[dim] = max(1, shape[dim] // 2)

        if dtype in (torch.int16, torch.int32, torch.int64):
            src = torch.randint(-8, 8, src_shape, device=device).to(dtype)
        else:
            src = torch.randn(src_shape, dtype=dtype, device=device)

        index = torch.randint(0, shape[dim], src_shape, dtype=torch.long, device=device)
        yield inp, dim, index, src, {"reduce": reduce, "include_self": include_self}

    return inner


def scatter_reduce_out_input_fn_factory(reduce="sum", include_self=True):
    def inner(shape, dtype, device):
        for inp, dim, index, src, kwargs in scatter_reduce_input_fn_factory(
            reduce, include_self
        )(shape, dtype, device):
            out = torch.empty_like(inp)
            yield inp, dim, index, src, kwargs, {"out": out}

    return inner


def scatter_reduce_gpu_dtypes(reduce):
    if reduce in ("sum", "mean"):
        return [torch.float16, torch.float32, torch.bfloat16, torch.int32]
    return [torch.float16, torch.float32, torch.bfloat16, torch.int32]


FORWARD_CASES = [
    ("scatter_reduce.forward.sum", "sum", True),
    ("scatter_reduce.forward.mean", "mean", False),
    ("scatter_reduce.forward.prod", "prod", True),
    ("scatter_reduce.forward.amax", "amax", False),
    ("scatter_reduce.forward.amin", "amin", True),
]

INPLACE_CASES = [
    ("scatter_reduce_.sum", "sum", True),
    ("scatter_reduce_.prod", "prod", False),
    ("scatter_reduce_.amax", "amax", False),
]

OUT_CASES = [
    ("scatter_reduce.out.mean", "mean", True),
    ("scatter_reduce.out.prod", "prod", True),
    ("scatter_reduce.out.amin", "amin", False),
]


@pytest.mark.scatter_reduce
@pytest.mark.parametrize(
    "op_name, reduce, include_self",
    FORWARD_CASES,
)
def test_perf_scatter_reduce(op_name, reduce, include_self):
    bench = ScatterReduceBenchmark(
        op_name=op_name,
        torch_op=torch.scatter_reduce,
        input_fn=scatter_reduce_input_fn_factory(reduce, include_self),
        get_gbps=scatter_reduce_gbps,
        dtypes=scatter_reduce_gpu_dtypes(reduce),
    )
    bench.run()


@pytest.mark.scatter_reduce
@pytest.mark.parametrize(
    "op_name, reduce, include_self",
    INPLACE_CASES,
)
def test_perf_scatter_reduce_inplace(op_name, reduce, include_self):
    bench = ScatterReduceBenchmark(
        op_name=op_name,
        torch_op=torch.Tensor.scatter_reduce_,
        input_fn=scatter_reduce_input_fn_factory(reduce, include_self),
        get_gbps=scatter_reduce_gbps,
        dtypes=scatter_reduce_gpu_dtypes(reduce),
        is_inplace=True,
    )
    bench.run()


@pytest.mark.scatter_reduce
@pytest.mark.parametrize(
    "op_name, reduce, include_self",
    OUT_CASES,
)
def test_perf_scatter_reduce_out(op_name, reduce, include_self):
    bench = ScatterReduceBenchmark(
        op_name=op_name,
        torch_op=torch.scatter_reduce,
        input_fn=scatter_reduce_out_input_fn_factory(reduce, include_self),
        get_gbps=scatter_reduce_gbps,
        dtypes=scatter_reduce_gpu_dtypes(reduce),
    )
    bench.run()
