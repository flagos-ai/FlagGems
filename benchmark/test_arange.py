# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

import flag_gems

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
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
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
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_arange_start_step():
    bench = base.GenericBenchmark(
        op_name="arange_start_step",
        input_fn=_input_fn_start_step,
        torch_op=torch.arange,
    )
    bench.run()
