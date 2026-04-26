import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


def _input_fn(shape, cur_dtype, device):
    inp1 = base.generate_tensor_input(shape, cur_dtype, device)
    inp2 = base.generate_tensor_input(shape, cur_dtype, device)

    yield inp1, inp2

    if base.Config.bench_level == attrs.BenchLevel.COMPREHENSIVE:
        # scalar situation
        yield inp1, 3.14


@pytest.mark.clamp_min
def test_clamp_min():
    bench = base.GenericBenchmark(
        op_name="clamp_min",
        input_fn=_input_fn,
        torch_op=torch.clamp_min,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clamp_min_
def test_clamp_min_inplace():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="clamp_min_",
        torch_op=torch.clamp_min_,
        dtypes=attrs.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
