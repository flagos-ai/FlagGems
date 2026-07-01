import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.conj
def test_conj():
    bench = base.UnaryPointwiseBenchmark(
        op_name="conj",
        torch_op=torch.conj,
        dtypes=consts.COMPLEX_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conj)
    bench.run()
