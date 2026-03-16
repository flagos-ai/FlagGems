import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, INT_DTYPES
from benchmark.performance_utils import GenericBenchmark2DOnly, generate_tensor_input


def tril_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    yield inp, 0


def tril_inplace_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    # yield a clone so each benchmark rep starts from the original data
    yield inp.clone(), 0


@pytest.mark.tril
def test_perf_tril():
    bench = GenericBenchmark2DOnly(
        op_name="tril",
        torch_op=torch.tril,
        input_fn=tril_input_fn,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
    )
    bench.run()


@pytest.mark.tril_
def test_perf_tril_inplace():
    bench = GenericBenchmark2DOnly(
        op_name="tril_",
        torch_op=torch.Tensor.tril_,
        input_fn=tril_inplace_input_fn,
        dtypes=FLOAT_DTYPES + INT_DTYPES,
        is_inplace=True,
    )
    bench.run()
