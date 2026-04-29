import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def smooth_l1_loss_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, target

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        yield inp, target, {"reduction": "mean"}
        yield inp, target, {"reduction": "sum"}
        yield inp, target, {"reduction": "none"}


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss():
    bench = utils.GenericBenchmark2DOnly(
        op_name="smooth_l1_loss",
        input_fn=smooth_l1_loss_input_fn,
        torch_op=torch.nn.functional.smooth_l1_loss,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
