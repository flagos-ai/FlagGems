import pytest
import torch

from . import base, consts


def _input_fn(shape, cur_dtype, device):
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0], device=device, dtype=cur_dtype)
    yield inp, boundaries


@pytest.mark.bucketize
def test_bucketize_perf():
    bench = base.GenericBenchmark(
        op_name="bucketize",
        input_fn=_input_fn,
        torch_op=torch.bucketize,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
