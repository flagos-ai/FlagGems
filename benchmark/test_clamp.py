import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp3 = utils.generate_tensor_input(shape, cur_dtype, device)

    yield inp1, inp2, inp3

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
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


def _clamp_tensor_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    mini = utils.generate_tensor_input(shape, cur_dtype, device)
    maxi = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, {"min": mini, "max": maxi}

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, {"min": mini, "max": None}
        yield inp, {"min": None, "max": maxi}


@pytest.mark.clamp_tensor
def test_clamp_tensor():
    bench = base.GenericBenchmark(
        op_name="clamp_tensor",
        input_fn=_clamp_tensor_input_fn,
        torch_op=torch.clamp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clamp_tensor_
def test_clamp_tensor_inplace():
    bench = base.GenericBenchmark(
        op_name="clamp_tensor_",
        input_fn=_clamp_tensor_input_fn,
        torch_op=torch.clamp_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
