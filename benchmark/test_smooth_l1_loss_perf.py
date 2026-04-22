import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, BenchLevel
from benchmark.performance_utils import Config, GenericBenchmark2DOnly, generate_tensor_input


def smooth_l1_loss_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    target = generate_tensor_input(shape, cur_dtype, device)
    yield inp, target
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        for reduction in ["mean", "sum", "none"]:
            for beta in [1.0, 0.1, 0.0]:
                yield inp, target, {"reduction": reduction, "beta": beta}


@pytest.mark.smooth_l1_loss
def test_perf_smooth_l1_loss():
    bench = GenericBenchmark2DOnly(
        input_fn=smooth_l1_loss_input_fn,
        op_name="smooth_l1_loss",
        torch_op=torch.nn.functional.smooth_l1_loss,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.smooth_l1_loss
def test_perf_smooth_l1_loss_backward():
    bench = GenericBenchmark2DOnly(
        input_fn=smooth_l1_loss_input_fn,
        op_name="smooth_l1_loss",
        torch_op=torch.nn.functional.smooth_l1_loss,
        dtypes=FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
