import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp3 = utils.generate_tensor_input(shape, cur_dtype, device)

    yield inp1, inp2, inp3

    if base.Config.bench_level == base.BenchLevel.COMPREHENSIVE:
        # scalar or None situation
        yield inp1, inp2, None
        yield inp1, None, 3.14


@pytest.mark.clamp
def test_clamp():
    bench = base.GenericBenchmark(
        op_name="clamp",
        input_fn=_input_fn,
        torch_op=torch.clamp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clamp_
def test_clamp_inplace():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="clamp_",
        torch_op=torch.clamp_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
