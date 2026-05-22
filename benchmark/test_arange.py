import math

import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    yield {
        "end": math.prod(shape),
        "device": device,
        "dtype": dtype,
    },

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield {
            "start": 0,
            "end": math.prod(shape),
            "step": 2,
            "device": device,
            "dtype": dtype,
        },


@pytest.mark.arange
def test_arange():
    bench = base.GenericBenchmark(
        op_name="arange", input_fn=_input_fn, torch_op=torch.arange
    )
    bench.run()


def _input_fn_arange_start(shape, dtype, device):
    yield {
        "start": 0,
        "end": math.prod(shape),
        "device": device,
        "dtype": dtype,
    },

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield {
            "start": 0,
            "end": math.prod(shape),
            "step": 2,
            "device": device,
            "dtype": dtype,
        },


@pytest.mark.arange_start
def test_arange_start():
    bench = base.GenericBenchmark(
        op_name="arange_start", input_fn=_input_fn_arange_start, torch_op=torch.arange
    )
    bench.run()


def _input_fn_start_step(shape, dtype, device):
    yield {
        "start": 0,
        "end": math.prod(shape),
        "step": 1,
        "device": device,
        "dtype": dtype,
    },

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield {
            "start": 10,
            "end": math.prod(shape) + 10,
            "step": 2,
            "device": device,
            "dtype": dtype,
        },


@pytest.mark.arange_start_step
def test_arange_start_step():
    bench = base.GenericBenchmark(
        op_name="arange_start_step",
        input_fn=_input_fn_start_step,
        torch_op=torch.arange,
    )
    bench.run()
