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


@pytest.mark.special_round_out
def test_special_round_out():
    # UnaryPointwiseOutBenchmark passes out= via kwargs internally
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_round_out",
        torch_op=torch.special.round,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
