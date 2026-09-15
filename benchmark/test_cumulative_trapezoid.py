import pytest
import torch

from . import base, consts, utils


def input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    # dim defaults to -1 in torch.cumulative_trapezoid; yield the tensor alone so
    # the benchmark records a bare shape detail.
    yield inp,


@pytest.mark.cumulative_trapezoid
def test_cumulative_trapezoid():
    bench = base.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="cumulative_trapezoid",
        torch_op=torch.cumulative_trapezoid,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
