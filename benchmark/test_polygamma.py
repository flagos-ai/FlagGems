import pytest
import torch

from . import base, consts


@pytest.mark.polygamma
def test_polygamma():
    bench = base.UnaryPointwiseBenchmark(
        op_name="polygamma",
        torch_op=lambda a: torch.polygamma(1, a),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def test_polygamma_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="polygamma_",
        torch_op=lambda a: a.polygamma_(1),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def test_polygamma_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="polygamma_out",
        torch_op=lambda a, out: torch.polygamma(1, a, out=out),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
