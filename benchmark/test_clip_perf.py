import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.conftest import BenchLevel, Config
from benchmark.performance_utils import GenericBenchmark, generate_tensor_input


def clip_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    yield inp, -0.5, 0.5
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield inp, None, 0.5
        yield inp, -0.5, None


@pytest.mark.clip
def test_clip():
    bench = GenericBenchmark(
        input_fn=clip_input_fn,
        op_name="clip",
        torch_op=torch.clip,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clip_
def test_clip_inplace():
    bench = GenericBenchmark(
        input_fn=clip_input_fn,
        op_name="clip_",
        torch_op=torch.clip_,
        dtypes=FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
