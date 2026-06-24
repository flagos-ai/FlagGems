import pytest
import torch

from . import base, consts


@pytest.mark.special_round
def test_special_round():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_round",
        torch_op=torch.special.round,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# torch.special.round_ does not exist natively; benchmark the in-place
# path via torch.special.round(x, out=x) which exercises the same kernel.
@pytest.mark.special_round_
def test_special_round_():
    def _round_inplace(x):
        return torch.special.round(x, out=x)

    bench = base.UnaryPointwiseBenchmark(
        op_name="special_round_",
        torch_op=_round_inplace,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_round_out
def test_special_round_out():
    # UnaryPointwiseOutBenchmark passes out= via kwargs internally
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_round_out",
        torch_op=torch.special.round,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
