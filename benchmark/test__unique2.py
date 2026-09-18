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


class Unique2Benchmark(base.GenericBenchmark):
    # Override DEFAULT_SHAPES to avoid exceeding sort's 2^30 limit
    # Keep shapes well under the limit for safety
    DEFAULT_SHAPES = [
        (1024,),
        (4096,),
        (16384,),
        (65536,),
        (262144,),
    ]

    def set_more_shapes(self):
        # Return empty since we already set all shapes in DEFAULT_SHAPES
        return []


def _input_fn_basic(shape, dtype, device):
    # Generate input with some repeated values for realistic unique operation
    if dtype in consts.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
        # Quantize to create duplicates
        inp = torch.round(inp * 10) / 10
    else:
        # For integer types, use a small range to ensure duplicates
        high = min(1000, shape[0] // 10) if shape[0] > 100 else 10
        inp = torch.randint(0, high, shape, dtype=dtype, device=device)
    yield inp, {"sorted": True, "return_inverse": False, "return_counts": False}


def _input_fn_return_inverse(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
        inp = torch.round(inp * 10) / 10
    else:
        high = min(1000, shape[0] // 10) if shape[0] > 100 else 10
        inp = torch.randint(0, high, shape, dtype=dtype, device=device)
    yield inp, {"sorted": True, "return_inverse": True, "return_counts": False}


def _input_fn_return_counts(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
        inp = torch.round(inp * 10) / 10
    else:
        high = min(1000, shape[0] // 10) if shape[0] > 100 else 10
        inp = torch.randint(0, high, shape, dtype=dtype, device=device)
    yield inp, {"sorted": True, "return_inverse": False, "return_counts": True}


def _input_fn_return_inverse_counts(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
        inp = torch.round(inp * 10) / 10
    else:
        high = min(1000, shape[0] // 10) if shape[0] > 100 else 10
        inp = torch.randint(0, high, shape, dtype=dtype, device=device)
    yield inp, {"sorted": True, "return_inverse": True, "return_counts": True}


@pytest.mark.unique2
def test_perf__unique2():
    bench = Unique2Benchmark(
        input_fn=_input_fn_basic,
        op_name="_unique2",
        torch_op=torch._unique2,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.unique2_return_inverse
def test_perf__unique2_return_inverse():
    bench = Unique2Benchmark(
        input_fn=_input_fn_return_inverse,
        op_name="_unique2_return_inverse",
        torch_op=torch._unique2,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.unique2_return_counts
def test_perf__unique2_return_counts():
    bench = Unique2Benchmark(
        input_fn=_input_fn_return_counts,
        op_name="_unique2_return_counts",
        torch_op=torch._unique2,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.unique2_return_inverse_counts
def test_perf__unique2_return_inverse_counts():
    bench = Unique2Benchmark(
        input_fn=_input_fn_return_inverse_counts,
        op_name="_unique2_return_inverse_counts",
        torch_op=torch._unique2,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()
