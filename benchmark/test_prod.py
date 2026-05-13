import pytest
import torch

from . import base, consts


@pytest.mark.prod
def test_prod():
    bench = base.UnaryReductionBenchmark(
        op_name="prod", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.prod_dim_int
def test_prod_dim_int():
    bench = base.UnaryReductionBenchmark(
        op_name="prod_dim_int", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
