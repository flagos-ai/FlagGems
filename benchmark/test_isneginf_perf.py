import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark, generate_tensor_input


def isneginf_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    inp = torch.masked_fill(inp, inp > 1.0, -float("inf"))
    yield inp,


@pytest.mark.isneginf
def test_perf_isneginf():
    bench = GenericBenchmark(
        input_fn=isneginf_input_fn,
        op_name="isneginf",
        torch_op=torch.isneginf,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
