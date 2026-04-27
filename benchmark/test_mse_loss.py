import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def mse_loss_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    target = utils.generate_tensor_input(shape, dtype, device)
    yield inp, target

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, target, {"reduction": "mean"}
        yield inp, target, {"reduction": "sum"}
        yield inp, target, {"reduction": "none"}


@pytest.mark.mse_loss
def test_mse_loss():
    bench = base.GenericBenchmark2DOnly(
        op_name="mse_loss",
        input_fn=mse_loss_input_fn,
        torch_op=torch.nn.functional.mse_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
