import pytest
import torch

from . import base, consts, utils


class ArgsortBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return [(1024, 1), (1024, 512), (16, 128 * 1024), (8, 256 * 1024)]


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1, "descending": False},


@pytest.mark.argsort
def test_argsort():
    bench = ArgsortBenchmark(
        input_fn=_input_fn,
        op_name="argsort",
        torch_op=torch.argsort,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()
