import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = (torch.randint(0, 2, shape, device=device).to(dtype) * 2) - 1
    yield inp, target


@pytest.mark.soft_margin_loss
def test_soft_margin_loss():
    bench = utils.GenericBenchmark(
        input_fn=_input_fn,
        op_name="soft_margin_loss",
        torch_op=torch.ops.aten.soft_margin_loss,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
