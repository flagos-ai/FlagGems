import pytest
import torch

from . import base, consts


@pytest.mark.sub
def test_sub():
    bench = base.BinaryPointwiseBenchmark(
        op_name="sub",
        torch_op=torch.sub,
        dtypes=consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.run()


# TODO(Qiming): Check why we don't have complex type here
@pytest.mark.sub_
def test_sub_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="sub_",
        torch_op=lambda a, b: a.sub_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
