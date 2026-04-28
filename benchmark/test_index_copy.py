import pytest
import torch

from benchmark.performance_utils import GenericBenchmark2DOnly
from flag_gems.utils import shape_utils


def index_copy_gbps(bench_fn_args, latency):
    index = bench_fn_args[2]
    src = bench_fn_args[3]
    io_amount = sum([shape_utils.size_in_bytes(item) for item in [index, src, src]])
    return io_amount * 1e-9 / (latency * 1e-3)


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
    bench = GenericBenchmark2DOnly(
        op_name="index_copy",
        torch_op=torch.index_copy,
        input_fn=index_copy_input_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
        get_gbps=index_copy_gbps,
    )
    bench.run()


@pytest.mark.index_copy_
def test_index_copy_():
    bench = GenericBenchmark2DOnly(
        op_name="index_copy_",
        torch_op=torch.Tensor.index_copy_,
        input_fn=index_copy_input_fn,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
        get_gbps=index_copy_gbps,
        inplace=True,
    )
    bench.run()
