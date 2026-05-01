import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)
    yield inp, target


@pytest.mark.smooth_l1_loss
def test_perf_smooth_l1_loss():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="smooth_l1_loss",
        torch_op=lambda inp, target: torch.nn.functional.smooth_l1_loss(
            inp, target, reduction="mean", beta=1.0
        ),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
