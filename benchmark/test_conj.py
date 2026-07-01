import pytest
import torch

import flag_gems

from . import base, consts

@pytest.mark.conj
def test_conj():
    dtypes = consts.COMPLEX_DTYPES + consts.FLOAT_DTYPES

    bench = base.UnaryPointwiseBenchmark(
        op_name="conj",
        torch_op=torch.conj,
        dtypes=dtypes,
    )

    bench.set_gems(flag_gems.conj)
    bench.run()
