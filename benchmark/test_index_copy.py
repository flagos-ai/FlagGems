import pytest
import torch

from . import base, consts, utils


class IndexCopyBenchmark(base.GenericBenchmark):
    def get_gbps(self, bench_fn_args, latency):
        index = bench_fn_args[2]
        src = bench_fn_args[3]
        io_amount = sum([utils.size_in_bytes(item) for item in [index, src, src]])
        return io_amount * 1e-9 / (latency * 1e-3)


def _tensor_input_fn(shape, dtype, device):
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
    bench = IndexCopyBenchmark(
        input_fn=_tensor_input_fn,
        op_name="index_copy",
        torch_op=torch.index_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.index_copy_
def test_index_copy_():
    bench = IndexCopyBenchmark(
        input_fn=_tensor_input_fn,
        op_name="index_copy_",
        torch_op=torch.Tensor.index_copy_,
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()
