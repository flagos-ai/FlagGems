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

import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    if shape[0] >= 819200:
        # Skip large shapes for performance testing
        return None

    if isinstance(shape, int):
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

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        for i in range(8, 13):
            n = 2**i
            m = 2**i
            yield {"n": n, "m": m, "dtype": dtype, "device": device},


@pytest.mark.eye
def test_eye():
    bench = base.GenericBenchmark(op_name="eye", input_fn=_input_fn, torch_op=torch.eye)
    bench.run()


def _input_fn_eye_m(shape, dtype, device):
    if shape[0] >= 819200:
        return None

    if isinstance(shape, int):
        yield {"n": shape, "m": shape, "dtype": dtype, "device": device},

    elif isinstance(shape, tuple) and len(shape) == 1:
        n = shape[0]
        yield {"n": n, "m": n, "dtype": dtype, "device": device},

    elif isinstance(shape, tuple) and len(shape) == 2:
        n, m = shape
        yield {"n": n, "m": m, "dtype": dtype, "device": device},

    elif isinstance(shape, tuple) and len(shape) > 2:
        n, m = shape[:2]
        yield {"n": n, "m": m, "dtype": dtype, "device": device},

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        for i in range(8, 13):
            n = 2**i
            m = 2**i
            yield {"n": n, "m": m, "dtype": dtype, "device": device},


@pytest.mark.eye_m
def test_eye_m():
    bench = base.GenericBenchmark(
        op_name="eye_m", input_fn=_input_fn_eye_m, torch_op=torch.eye
    )
    bench.run()
