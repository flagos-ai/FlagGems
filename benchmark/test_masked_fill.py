import pytest
import torch

from . import base, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    mask = utils.generate_tensor_input(shape, dtype, device) < 0.3
    value = 1024

    yield inp, mask, value


@pytest.mark.masked_fill
def test_masked_fill():
    bench = base.GenericBenchmark(
        op_name="masked_fill", input_fn=_input_fn, torch_op=torch.masked_fill
    )
    bench.run()


@pytest.mark.masked_fill_
def test_masked_fill_inplace():
    bench = base.GenericBenchmark(
        op_name="masked_fill_",
        input_fn=_input_fn,
        torch_op=lambda a, b, c: a.masked_fill_(b, c),
        is_inplace=True,
    )

    bench.run()
