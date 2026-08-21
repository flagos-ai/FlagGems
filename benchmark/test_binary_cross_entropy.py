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


def binary_cross_entropy_out_input_fn(shape, dtype, device):
    # Generate input in (0, 1) range using sigmoid
    inp = torch.sigmoid(utils.generate_tensor_input(shape, dtype, device))
    # Generate binary targets (0 or 1)
    target = torch.randint(0, 2, shape, device=device).to(dtype)

    # Default reduction is "mean", so out is scalar
    out = torch.empty((), dtype=dtype, device=device)
    yield inp, target, {"out": out}

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # reduction="mean": out is scalar
        yield inp, target, {
            "reduction": "mean",
            "out": torch.empty((), dtype=dtype, device=device),
        }

        # reduction="sum": out is scalar
        yield inp, target, {
            "reduction": "sum",
            "out": torch.empty((), dtype=dtype, device=device),
        }

        # reduction="none": out shape matches input shape
        yield inp, target, {
            "reduction": "none",
            "out": torch.empty_like(inp),
        }


def torch_binary_cross_entropy_out(inp, target, **kwargs):
    out = kwargs.pop("out")
    result = torch.nn.functional.binary_cross_entropy(inp, target, **kwargs)
    out.copy_(result)
    return out


@pytest.mark.binary_cross_entropy
def test_binary_cross_entropy():
    bench = base.GenericBenchmark2DOnly(
        op_name="binary_cross_entropy",
        input_fn=binary_cross_entropy_input_fn,
        torch_op=torch.nn.functional.binary_cross_entropy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.binary_cross_entropy_out
def test_binary_cross_entropy_out():
    bench = base.GenericBenchmark2DOnly(
        op_name="binary_cross_entropy_out",
        input_fn=binary_cross_entropy_out_input_fn,
        torch_op=torch_binary_cross_entropy_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
