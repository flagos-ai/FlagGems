import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def nll_loss_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target_shape = list(shape)
    del target_shape[1]
    target = torch.randint(0, shape[-1], target_shape, device=device)
    yield inp, target

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}


@pytest.mark.nll_loss2d
def test_nll_loss2d():
    bench = utils.GenericBenchmark4DOnly(
        input_fn=nll_loss_input_fn,
        op_name="nll_loss2d",
        torch_op=torch.nn.functional.nll_loss,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
