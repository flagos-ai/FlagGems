import pytest
import torch

from . import base


@pytest.mark.special_spherical_bessel_j0
def test_special_spherical_bessel_j0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_spherical_bessel_j0",
        torch_op=torch.special.spherical_bessel_j0,
        # torch.special.spherical_bessel_j0 does not support fp16/bf16
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.special_spherical_bessel_j0_
def test_special_spherical_bessel_j0_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_spherical_bessel_j0_",
        torch_op=torch.special.spherical_bessel_j0_,
        # torch.special.spherical_bessel_j0 does not support fp16/bf16
        dtypes=[torch.float32],
    )
    bench.run()
