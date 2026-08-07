import pytest
import torch

from . import base, consts


@pytest.mark.special_psi
def test_special_psi():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_psi",
        torch_op=torch.special.psi,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
