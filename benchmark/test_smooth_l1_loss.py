import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, BenchLevel
from benchmark.performance_utils import Config, GenericBenchmark, generate_tensor_input


def smooth_l1_loss_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = generate_tensor_input(shape, cur_dtype, device)
    yield inp, target
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield inp, target, {"reduction": "mean", "beta": 1.0}
        yield inp, target, {"reduction": "sum", "beta": 1.0}
        yield inp, target, {"reduction": "none", "beta": 1.0}


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "smooth_l1_loss",
            torch.nn.functional.smooth_l1_loss,
            smooth_l1_loss_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.smooth_l1_loss,
        ),
    ],
)
def test_perf_smooth_l1_loss(op_name, torch_op, input_fn, dtypes):
    bench = GenericBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
    )
    bench.run()
