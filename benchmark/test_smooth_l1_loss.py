import pytest
import torch

from . import base, consts, utils


def smooth_l1_loss_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    target = utils.generate_tensor_input(shape, dtype, device)
    yield inp, target

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, target, {"reduction": "mean", "beta": 1.0}
        yield inp, target, {"reduction": "sum", "beta": 1.0}
        yield inp, target, {"reduction": "none", "beta": 1.0}


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss():
    bench = base.GenericBenchmark2DOnly(
        op_name="smooth_l1_loss",
        input_fn=smooth_l1_loss_input_fn,
        torch_op=torch.nn.functional.smooth_l1_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
