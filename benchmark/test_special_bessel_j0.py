import pytest
import torch

from . import base


@pytest.mark.special_bessel_j0
def test_special_bessel_j0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_bessel_j0",
        torch_op=torch.special.bessel_j0,
        # torch.special.bessel_j0 does not support fp16/bf16
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_bessel_j0_
def test_special_bessel_j0_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_bessel_j0_",
        torch_op=lambda x: x.copy_(torch.special.bessel_j0(x)),
        # torch.special.bessel_j0 does not support fp16/bf16
        dtypes=[torch.float32],
        is_inplace=True,
    )
    bench.run()
