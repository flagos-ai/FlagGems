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

from . import base, consts, utils


def input_fn(shape, dtype, device):
    # Use well-conditioned input in [0.75, 1.25] rather than randn(). A running
    # product of standard-normal values underflows to exactly 0.0 within a few
    # dozen steps, so for long reduction axes (e.g. 4096 or 65536) most of the
    # forward output becomes zero. aten::cumprod_backward has a global branch
    # that abandons its fast vectorized path and runs a serial per-line
    # zero-handling routine whenever any zero is present, which makes the eager
    # baseline pathologically slow (hundreds to thousands of ms) and inflates
    # the reported speedup into a meaningless artifact. Keeping the cumulative
    # product O(1) exercises both implementations on their fast paths and yields
    # a fair, reproducible comparison. Zero-handling correctness is covered by
    # tests/test_cumprod.py::test_cumprod_backward.
    inp = torch.rand(shape, dtype=dtype, device=device) * 0.5 + 0.75
    grad = utils.generate_tensor_input(shape, dtype, device)
    output = torch.cumprod(inp, dim=1)
    yield grad, inp, 1, output


@pytest.mark.cumprod_backward
def test_cumprod_backward():
    bench = base.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="cumprod_backward",
        torch_op=torch.ops.aten.cumprod_backward,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
