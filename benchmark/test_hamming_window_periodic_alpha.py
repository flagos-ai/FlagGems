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
        "window_length": math.prod(shape),
        "periodic": True,
        "alpha": 0.54,
        "dtype": dtype,
        "device": device,
    },

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield {
            "window_length": math.prod(shape),
            "periodic": False,
            "alpha": 0.54,
            "dtype": dtype,
            "device": device,
        },


def _input_fn_beta(shape, dtype, device):
    yield {
        "window_length": math.prod(shape),
        "periodic": True,
        "alpha": 0.54,
        "beta": 0.46,
        "dtype": dtype,
        "device": device,
    },

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield {
            "window_length": math.prod(shape),
            "periodic": False,
            "alpha": 0.54,
            "beta": 0.46,
            "dtype": dtype,
            "device": device,
        },


@pytest.mark.hamming_window_periodic_alpha
def test_hamming_window_periodic_alpha():
    bench = base.GenericBenchmark(
        op_name="hamming_window_periodic_alpha",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.hamming_window.periodic_alpha,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.hamming_window_periodic_alpha_beta
def test_hamming_window_periodic_alpha_beta():
    bench = base.GenericBenchmark(
        op_name="hamming_window_periodic_alpha_beta",
        input_fn=_input_fn_beta,
        torch_op=torch.ops.aten.hamming_window.periodic_alpha_beta,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
