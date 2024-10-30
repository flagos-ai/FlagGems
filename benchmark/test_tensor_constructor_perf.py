import math

import pytest
import torch

from .attri_util import BenchLevel
from .performance_utils import (
    Config,
    GenericBenchmark,
    generate_tensor_input,
    unary_input_fn,
)


def generic_constructor_input_fn(shape, dtype, device):
    yield {"size": shape, "dtype": dtype, "device": device},


def full_input_fn(shape, dtype, device):
    yield {"size": shape, "fill_value": 3.1415926, "dtype": dtype, "device": device},


def masked_fill_input_fn(shape, dtype, device):
    inp = generate_tensor_input(shape, dtype, device)
    mask = generate_tensor_input(shape, dtype, device) < 0.3
    value = 1024
    yield inp, mask, value


def full_like_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield {"input": inp, "fill_value": 3.1415926},


def fill_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    yield input, 3.14159,


def arange_input_fn(shape, dtype, device):
    yield {
        "end": math.prod(shape),
        "device": device,
        "dtype": dtype,
    },
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield {
            "start": 0,
            "end": math.prod(shape),
            "step": 2,
            "device": device,
            "dtype": dtype,
        },


# Define operations and their corresponding input functions
tensor_constructor_operations = [
    # generic tensor constructor
    ("rand", torch.rand, generic_constructor_input_fn),
    ("randn", torch.randn, generic_constructor_input_fn),
    ("ones", torch.ones, generic_constructor_input_fn),
    ("zeros", torch.zeros, generic_constructor_input_fn),
    # generic tensor-like constructor
    ("rand_like", torch.rand_like, unary_input_fn),
    ("randn_like", torch.randn_like, unary_input_fn),
    ("ones_like", torch.ones_like, unary_input_fn),
    ("zeros_like", torch.zeros_like, unary_input_fn),
    # tensor constructor with given value
    ("fill", torch.fill, fill_input_fn),
    ("masked_fill", torch.masked_fill, masked_fill_input_fn),
    ("full", torch.full, full_input_fn),
    ("full_like", torch.full_like, full_like_input_fn),
    # arange
    ("arange", torch.arange, arange_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(op, fn, input_fn, marks=getattr(pytest.mark, op, None))
        for op, fn, input_fn in tensor_constructor_operations
    ],
)
def test_tensor_constructor_benchmark(op_name, torch_op, input_fn):
    bench = GenericBenchmark(input_fn=input_fn, op_name=op_name, torch_op=torch_op)
    bench.run()


# @pytest.mark.randperm
# def test_perf_randperm():
#     def randperm_input_fn(shape, dtype, device):
#         yield {"n": shape[0], "dtype": dtype, "device": device},

#     bench = GenericBenchmark(
#         input_fn=randperm_input_fn,
#         op_name="randperm",
#         torch_op=torch.randperm,
#         dtypes=[torch.int32, torch.int64],
#     )
#     bench.run()
