import pytest
import torch

from . import base, consts


def binary_cross_entropy_input_fn(shape, cur_dtype, device):
    inp = torch.rand(shape, dtype=cur_dtype, device=device)
    target = torch.rand(shape, dtype=cur_dtype, device=device)
    yield inp, target


@pytest.mark.binary_cross_entropy
def test_perf_binary_cross_entropy():
    bench = base.GenericBenchmark(
        input_fn=binary_cross_entropy_input_fn,
        op_name="binary_cross_entropy",
        torch_op=torch.nn.functional.binary_cross_entropy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
