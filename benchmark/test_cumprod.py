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
from flag_gems.ops import cumprod as flag_gems_cumprod

from . import base, consts

CUMPROD_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.BOOL_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
)
CUMPROD_INPLACE_DTYPES = (
    consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.EXTRA_INT_DTYPES
)


def _make_input(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES:
        return torch.empty(shape, dtype=dtype, device=device).uniform_(0.99, 1.01)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.int8, device="cpu").to(
            device, dtype=dtype
        )
    if dtype is torch.uint8:
        return torch.randint(0, 4, shape, dtype=dtype, device="cpu").to(device)
    return torch.randint(-3, 4, shape, dtype=dtype, device="cpu").to(device)


def input_fn(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    yield inp, 1


# Ascend does not provide a stable native torch.cumprod(bool) baseline, while
# bool cumprod is equivalent to uint8 cumprod for 0/1 values. Keep this adapter
# limited to the benchmark baseline so the benchmark still compares against the
# vendor native implementation instead of the FlagGems wrapper.
def torch_cumprod(inp, dim):
    if flag_gems.vendor_name == "ascend" and inp.dtype is torch.bool:
        return torch.cumprod(inp.to(torch.uint8), dim)
    return torch.cumprod(inp, dim)


@pytest.mark.cumprod
def test_cumprod():
    bench = base.GenericBenchmark2DOnly(
        op_name="cumprod",
        input_fn=input_fn,
        torch_op=torch_cumprod,
        gems_op=flag_gems_cumprod,
        dtypes=CUMPROD_DTYPES,
    )
    bench.run()


@pytest.mark.cumprod_
def test_cumprod_():
    bench = base.GenericBenchmark2DOnly(
        op_name="cumprod_",
        input_fn=input_fn,
        torch_op=torch.Tensor.cumprod_,
        dtypes=CUMPROD_INPLACE_DTYPES,
        is_inplace=True,
    )
    bench.run()


def cumprod_backward_input_fn(shape, dtype, device):
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
    # The gradient must stay finite: with a NaN/Inf gradient the eager baseline
    # returns a NaN result, and timing a NaN-poisoned baseline against a real
    # kernel makes the reported speedup meaningless. utils.generate_tensor_input
    # samples normal * 10, which can overflow fp16, so draw a bounded gradient.
    grad = torch.randn(shape, dtype=dtype, device=device)
    output = torch.cumprod(inp, dim=1)
    yield grad, inp, 1, output


@pytest.mark.cumprod_backward
def test_cumprod_backward():
    bench = base.GenericBenchmark2DOnly(
        op_name="cumprod_backward",
        input_fn=cumprod_backward_input_fn,
        torch_op=torch.ops.aten.cumprod_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
