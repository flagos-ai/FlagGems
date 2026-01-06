import math
import os
import random

import pytest
import torch

import flag_gems
from benchmark.attri_util import BenchLevel
from benchmark.performance_utils import (
    Config,
    GenericBenchmark,
    SkipVersion,
    generate_tensor_input,
    unary_input_fn,
    vendor_name,
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


def linspace_input_fn(shape, dtype, device):
    limit = torch.finfo(dtype).max - 1
    num = int(min(limit, math.prod(shape)))
    yield {
        "start": 0,
        "end": num,
        "steps": random.randint(1, num),
        "dtype": dtype,
        "device": device,
    },


def logspace_input_fn(shape, dtype, device):
    base = 1.05
    limit = math.log2(torch.finfo(dtype).max - 1) / math.log2(
        base
    )  # calculate the max limit according to dtype
    end = int(limit)
    yield {
        "start": 0,
        "end": end,
        "steps": math.prod(shape),  # steps influence speed up a lot
        "base": base,
        "dtype": dtype,
        "device": device,
    },


def _2D_input_fn(shape, dtype, device):
    """
    Generate input for 2D input
    """
    if shape[0] >= 819200:
        # Skip large shapes for performance testing
        return
    elif isinstance(shape, int):
        yield {"n": shape, "dtype": dtype, "device": device},

    elif isinstance(shape, tuple) and len(shape) == 1:
        n = shape[0]
        yield {"n": n, "dtype": dtype, "device": device},

    elif isinstance(shape, tuple) and len(shape) == 2:
        n, m = shape
        yield {"n": n, "m": m, "dtype": dtype, "device": device},

    elif isinstance(shape, tuple) and len(shape) > 2:
        n, m = shape[:2]
        yield {"n": n, "m": m, "dtype": dtype, "device": device},
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        for i in range(8, 13):
            n = 2**i
            m = 2**i
            yield {"n": n, "m": m, "dtype": dtype, "device": device},


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
    # linspace
    ("linspace", torch.linspace, linspace_input_fn),
    # eye
    ("eye", torch.eye, _2D_input_fn),
    # logspace
    ("logspace", torch.logspace, logspace_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(op, fn, input_fn, marks=getattr(pytest.mark, op, None))
        for op, fn, input_fn in tensor_constructor_operations
    ],
)
def test_tensor_constructor_benchmark(op_name, torch_op, input_fn):
    if (
        vendor_name == "kunlunxin"
        and SkipVersion("torch", "<2.5")
        and op_name
        in [
            "linspace",
        ]
    ):
        pytest.skip("only support torch >= 2.5.")
    bench = GenericBenchmark(input_fn=input_fn, op_name=op_name, torch_op=torch_op)
    bench.run()


tensor_constructor_inplace_operations = [
    # tensor constructor with given value
    ("fill_", torch.fill_, fill_input_fn),
    ("masked_fill_", lambda a, b, c: a.masked_fill_(b, c), masked_fill_input_fn),
]


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(op, fn, input_fn, marks=getattr(pytest.mark, op, None))
        for op, fn, input_fn in tensor_constructor_inplace_operations
    ],
)
def test_tensor_constructor_inplace_benchmark(op_name, torch_op, input_fn):
    bench = GenericBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, is_inplace=True
    )
    bench.run()


# @pytest.mark.skipif(vendor_name == "hygon", reason="RESULT TODOFIX")
@pytest.mark.randperm
def test_perf_randperm():
    if flag_gems.vendor_name == "mthreads":
        os.environ["DISABLE_LLVM_OPT"] = "1"

    def randperm_input_fn(shape, dtype, device):
        yield {"n": shape[0], "dtype": dtype, "device": device},

    bench = GenericBenchmark(
        input_fn=randperm_input_fn,
        op_name="randperm",
        torch_op=torch.randperm,
        dtypes=[torch.int32, torch.int64],
    )
    bench.run()

    if flag_gems.vendor_name == "mthreads":
        del os.environ["DISABLE_LLVM_OPT"]
