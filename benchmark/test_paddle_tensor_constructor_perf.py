import math
import random

import pytest
import torch

from benchmark.attri_util import BenchLevel
from benchmark.performance_utils import (
    Config,
    GenericBenchmark,
    generate_tensor_input,
    unary_input_fn,
    vendor_name,
)


def generic_constructor_input_fn(shape, dtype, device):
    yield {"size": shape, "dtype": dtype, "device": device},

# use_gems() takes these three over by replacing torch.ones / zeros / zeros_like
# themselves (the paddle functions never reach _C_ops.ones and friends). Holding
# the function objects here would keep calling the originals inside use_gems(), so
# look the name up at call time, the way the other paddle benchmarks reach _C_ops.
def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)


def zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs)


# Define operations and their corresponding input functions
tensor_constructor_operations = [
    # generic tensor constructor
    ("ones", ones, generic_constructor_input_fn),
    ("zeros", zeros, generic_constructor_input_fn),
    # generic tensor-like constructor
    ("zeros_like", zeros_like, unary_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(op, fn, input_fn, marks=getattr(pytest.mark, op, None))
        for op, fn, input_fn in tensor_constructor_operations
    ],
)
def test_tensor_constructor_benchmark(op_name, torch_op, input_fn):
    if vendor_name == "kunlunxin" and op_name in [
        "linspace",
    ]:
        pytest.skip("RUNTIME TODOFIX.")
    if vendor_name == "mthreads" and op_name == "logspace":
        pytest.skip("Torch MUSA Unsupported Now")
    bench = GenericBenchmark(input_fn=input_fn, op_name=op_name, torch_op=torch_op)
    bench.run()

