import pytest
import torch

from . import base, consts


@pytest.mark.special_modified_bessel_i0
def test_special_modified_bessel_i0():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_modified_bessel_i0",
        torch_op=torch.i0,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_modified_bessel_i0_out
def test_special_modified_bessel_i0_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_modified_bessel_i0_out",
        torch_op=torch.special.modified_bessel_i0,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
