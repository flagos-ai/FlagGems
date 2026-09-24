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

from . import base, consts


def _input_fn(shape, dtype, device):
    yield {
        "window_length": math.prod(shape),
        "periodic": True,
        "beta": 12.0,
        "dtype": dtype,
        "device": device,
    },

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield {
            "window_length": math.prod(shape),
            "periodic": False,
            "beta": 12.0,
            "dtype": dtype,
            "device": device,
        },


@pytest.mark.kaiser_window
def test_kaiser_window():
    bench = base.GenericBenchmark(
        op_name="kaiser_window",
        input_fn=_input_fn,
        torch_op=torch.kaiser_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.kaiser_window_beta
def test_kaiser_window_beta():
    bench = base.GenericBenchmark(
        op_name="kaiser_window_beta",
        input_fn=_input_fn,
        torch_op=torch.kaiser_window,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
