import pytest
import torch

from . import base, consts


@pytest.mark.special_erfinv
def test_special_erfinv():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_erfinv",
        torch_op=torch.ops.aten.special_erfinv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_erfinv_out
def test_special_erfinv_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_erfinv_out",
        torch_op=torch.ops.aten.special_erfinv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
