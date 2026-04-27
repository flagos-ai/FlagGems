import pytest
import torch

from . import attri_util as consts
from . import performance_utils as base
from . import utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, -0.5, 0.5
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, None, 0.5
        yield inp, -0.5, None


@pytest.mark.clip
def test_clip():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="clip",
        torch_op=torch.clip,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clip_
def test_clip_inplace():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="clip_",
        torch_op=torch.clip_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
