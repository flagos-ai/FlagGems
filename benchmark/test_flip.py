import pytest
import torch

from . import base, consts


def _input_fn(shape, cur_dtype, device):
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    if len(shape) > 1:
        yield inp, {"dims": (0, 1)}
    else:
        yield inp, {"dims": (0,)}


@pytest.mark.flip
def test_flip():
    bench = base.GenericBenchmark(
        op_name="flip",
        input_fn=_input_fn,
        torch_op=torch.flip,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
