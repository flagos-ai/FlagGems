import pytest
import torch

import flag_gems

from . import attri_util as attrs
from . import performance_utils as base
from . import utils


def nll_loss_input_fn(shape, cur_dtype, device):
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    target_shape = list(shape)
    del target_shape[1]
    target = torch.randint(0, shape[-1], target_shape, device=device)
    yield inp, target

    if base.Config.bench_level == attrs.BenchLevel.COMPREHENSIVE:
        weight = torch.randn(shape[1], dtype=cur_dtype, device=device)
        yield inp, target, {"weight": weight, "ignore_index": 1, "reduction": "none"}


@pytest.mark.nll_loss
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin" and utils.SkipVersion("torch", "<2.5"),
    reason="INT16 is not supported in XPytorch 2.0. Please upgrade your PyTorch version >= 2.5",
)
def test_nll_loss():
    bench = base.GenericBenchmark2DOnly(
        op_name="nll_loss",
        input_fn=nll_loss_input_fn,
        torch_op=torch.nn.functional.nll_loss,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()
