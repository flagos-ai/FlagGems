import pytest
import torch

from . import base, consts, utils


def binary_cross_entropy_input_fn(shape, dtype, device):
    # Generate input in (0, 1) range using sigmoid
    inp = torch.sigmoid(utils.generate_tensor_input(shape, dtype, device))
    # Generate binary targets (0 or 1)
    target = torch.randint(0, 2, shape, device=device).to(dtype)
    yield inp, target

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, target, {"reduction": "mean"}
        yield inp, target, {"reduction": "sum"}
        yield inp, target, {"reduction": "none"}


@pytest.mark.binary_cross_entropy
def test_binary_cross_entropy():
    bench = base.GenericBenchmark2DOnly(
        op_name="binary_cross_entropy",
        input_fn=binary_cross_entropy_input_fn,
        torch_op=torch.nn.functional.binary_cross_entropy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.binary_cross_entropy
def test_binary_cross_entropy_out():
    bench = base.GenericBenchmark2DOnly(
        op_name="binary_cross_entropy",
        input_fn=binary_cross_entropy_input_fn,
        torch_op=torch.nn.functional.binary_cross_entropy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
