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

import flag_gems

from . import base, consts

vendor_name = flag_gems.vendor_name


class LerpBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return []


def lerp_input_fn(shape, dtype, device):
    input = torch.randn(*shape, device=device, dtype=dtype)
    end = input + 10
    weight = torch.randn(*shape, device=device, dtype=dtype)
    yield {"input": input, "end": end, "weight": weight},


@pytest.mark.lerp_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_lerp_tensor():
    bench = LerpBenchmark(
        input_fn=lerp_input_fn,
        op_name="lerp_tensor",
        torch_op=torch.lerp,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


@pytest.mark.lerp_tensor_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_lerp_tensor_inplace():
    bench = LerpBenchmark(
        input_fn=lerp_input_fn,
        op_name="lerp_tensor_",
        torch_op=lambda input, end, weight: input.lerp_(end, weight),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def lerp_scalar_input_fn(shape, dtype, device):
    input = torch.randn(*shape, device=device, dtype=dtype)
    end = input + 10
    yield {"input": input, "end": end, "weight": 0.5},


@pytest.mark.lerp_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_lerp_scalar():
    bench = LerpBenchmark(
        input_fn=lerp_scalar_input_fn,
        op_name="lerp_scalar",
        torch_op=torch.lerp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.lerp_scalar_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_lerp_scalar_inplace():
    bench = LerpBenchmark(
        input_fn=lerp_scalar_input_fn,
        op_name="lerp_scalar_",
        torch_op=lambda input, end, weight: input.lerp_(end, weight),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
