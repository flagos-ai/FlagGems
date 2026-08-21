import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    # alpha_dropout_ takes (input, p, train) — p defaults to 0.5
    yield inp, 0.5, True


@pytest.mark.alpha_dropout_
def test_alpha_dropout_():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="alpha_dropout_",
        torch_op=lambda inp, p, train: torch.alpha_dropout_(inp, p, train),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
