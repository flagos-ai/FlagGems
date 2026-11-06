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
from .conftest import Config


class HygonNonzeroAPIBenchmark(base.GenericBenchmark):
    def __init__(self, *args, shape, **kwargs):
        self.case_shape = shape
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [self.case_shape]


@pytest.mark.nonzero
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon override")
@pytest.mark.parametrize(
    "shape,density",
    [
        ((), 1.0),
        ((1023,), 0.0),
        ((1024,), 0.01),
        ((1025,), 1.0),
        ((33, 31), 0.5),
        ((2, 3, 5, 7), 0.5),
        ((8193,), 0.01),
        ((2**22,), 0.01),
        ((2**22 + 1,), 0.01),
    ],
)
def test_hygon_nonzero_full_api(shape, density):
    # Dynamic cardinality requires a host read. Device-event/kernel-only or
    # graph timing cannot describe its allocation + synchronization API cost.
    if Config.mode != consts.BenchMode.OPERATOR:
        pytest.skip("nonzero full API requires --mode operator")

    def input_fn(shape, dtype, device):
        generator = torch.Generator(device="cpu").manual_seed(20261104)
        inp = torch.randn(shape, dtype=dtype, generator=generator)
        inp *= torch.rand(shape, generator=generator) < density
        yield inp.to(device), {"as_tuple": False}

    bench = HygonNonzeroAPIBenchmark(
        shape=shape,
        input_fn=input_fn,
        op_name="nonzero",
        torch_op=torch.nonzero,
        gems_op=flag_gems.nonzero,
        dtypes=[torch.float32, torch.float16],
    )
    # Benchmark's OPERATOR mode warms up JIT, synchronizes before/after a wall
    # clock loop and calls the entire API each iteration, including total.item,
    # exact output allocation, scatter launch and final stream completion.
    bench.run()
