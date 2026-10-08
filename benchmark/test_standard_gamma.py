import pytest
import torch

from . import base, consts


@pytest.mark.standard_gamma
def test_standard_gamma():
    bench = base.UnaryPointwiseBenchmark(
        op_name="standard_gamma",
        torch_op=torch._standard_gamma,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
