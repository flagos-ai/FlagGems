import pytest
import torch

from . import base, consts


@pytest.mark.conj_physical
def test_conj_physical():
    bench = base.UnaryPointwiseBenchmark(
        op_name="conj_physical",
        torch_op=torch.conj_physical,
        dtypes=consts.COMPLEX_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()
