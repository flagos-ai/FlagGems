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


def _softshrink_reference(inp, lambd=0.5):
    """Device-side softshrink reference: x-lambd if x>lambd; x+lambd if x<-lambd; else 0.

    torch-xmlir's XDNN kernel has no bf16 `aten.softshrink` ([NOT
    IMPLEMENTED]: kbfloat16 unsupported in xdnn_pytorch_wrapper), so the eager
    reference is expressed with nested torch.where on device.
    """
    return torch.where(
        inp > lambd,
        inp - lambd,
        torch.where(inp < -lambd, inp + lambd, torch.zeros_like(inp)),
    )


def _softshrink_out_reference(inp, lambd=0.5, out=None):
    """Device-side softshrink.out reference, written into the caller-provided out buffer."""
    if out is None:
        out = torch.empty_like(inp)
    torch.where(
        inp > lambd,
        inp - lambd,
        torch.where(inp < -lambd, inp + lambd, torch.zeros_like(inp)),
        out=out,
    )
    return out


@pytest.mark.softshrink
def test_softshrink():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softshrink",
        torch_op=_softshrink_reference,
        gems_op=flag_gems.softshrink,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.softshrink_out
def test_softshrink_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="softshrink_out",
        torch_op=_softshrink_out_reference,
        gems_op=flag_gems.softshrink_out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
