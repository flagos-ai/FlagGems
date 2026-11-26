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

from . import base, consts, utils


class UniqueBenchmark(base.GenericBenchmark2DOnly):
    """Filter out overly large shapes that trigger 'out of resource: threads'."""

    MAX_ELEMENTS = 2**29  # 512M elements

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        return [s for s in shapes if math.prod(s) <= self.MAX_ELEMENTS]


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"sorted": True, "return_inverse": True, "return_counts": False},


@pytest.mark.unique2
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_unique2():
    bench = UniqueBenchmark(
        input_fn=_input_fn,
        op_name="unique2",
        torch_op=torch.unique,
        dtypes=consts.INT_DTYPES,
    )

    bench.run()
