import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    inp1 = torch.randn(shape, dtype=dtype, device=device)
    inp2 = torch.randn(shape, dtype=dtype, device=device)
    target = (torch.randint(0, 2, shape, device=device, dtype=torch.int8) * 2 - 1).to(
        dtype
    )
    yield inp1, inp2, target, 0.5, 1


@pytest.mark.margin_ranking_loss
def test_margin_ranking_loss():
    bench = utils.GenericBenchmark(
        op_name="margin_ranking_loss",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.margin_ranking_loss,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
