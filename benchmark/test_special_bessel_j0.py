import pytest
import torch

from . import base


@pytest.mark.special_bessel_j0
def test_special_bessel_j0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_bessel_j0",
        torch_op=torch.special.bessel_j0,
        # torch.special.bessel_j0 does not support fp16/bf16
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
