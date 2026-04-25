import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as utils


def cross_entropy_loss_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, shape[-1], (shape[0],), device=device)
    yield inp, target

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[-1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}
        yield inp, target, {
            "weight": weight,
            "reduction": "sum",
            "label_smoothing": 0.1,
        }


@pytest.mark.cross_entropy_loss
def test_cross_entropy_loss():
    bench = utils.GenericBenchmark2DOnly(
        input_fn=cross_entropy_loss_input_fn,
        op_name="cross_entropy_loss",
        torch_op=torch.nn.functional.cross_entropy,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.cross_entropy_loss)
    bench.run()
