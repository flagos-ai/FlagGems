import pytest
import torch

from . import base, consts


@pytest.mark.erf
def test_erf():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erf", torch_op=torch.erf, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.erf_
def test_erf_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="erf_", torch_op=torch.erf_, dtypes=consts.FLOAT_DTYPES, is_inplace=True
    )
    bench.run()
