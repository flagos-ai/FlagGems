import pytest
import torch

from . import base, consts


@pytest.mark.special_sinc
def test_special_sinc():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_sinc",
        torch_op=torch.ops.aten.special_sinc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_sinc_
def test_special_sinc_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_sinc_",
        torch_op=torch.ops.aten.sinc_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
