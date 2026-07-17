import pytest
import torch

from . import base, consts


@pytest.mark.conj
def test_conj():
    # _conj only operates on complex dtypes (FLOAT_DTYPES not applicable)
    bench = base.UnaryPointwiseBenchmark(
        op_name="conj",
        torch_op=torch._conj,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.run()
