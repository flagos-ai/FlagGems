import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.5, True


@pytest.mark.feature_dropout
def test_feature_dropout():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="feature_dropout",
        torch_op=torch.feature_dropout,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.feature_dropout_
def test_feature_dropout_():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="feature_dropout_",
        torch_op=torch.feature_dropout_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
