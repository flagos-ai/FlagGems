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

from . import base

vendor_name = flag_gems.vendor_name

# NOTE: This is a dead function identified during refactoring
# def weight_norm_interface_input_fn(shape, dtype, device):
#    dim = 0
#    v = torch.randn(shape, dtype=dtype, device=device)
#    g = torch.randn(shape[dim], dtype=dtype, device=device)
#    yield v, g, dim


def weight_norm_input_fn(shape, dtype, device):
    v = torch.randn(shape, dtype=dtype, device=device)
    if vendor_name in ["cambricon", "enflame"]:
        # Cambricon and Enflame fix input shape limit.
        g = torch.randn(shape[:1] + (1,) * (len(shape) - 1), dtype=dtype, device=device)
    else:
        g = torch.randn(shape, dtype=dtype, device=device)
    yield v, g, 0


@pytest.mark.weight_norm_interface
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="weight_norm_interface",
        input_fn=weight_norm_input_fn,
        torch_op=torch._weight_norm,
    )
    bench.set_gems(flag_gems.weight_norm)

    bench.run()


def weight_norm_interface_backward_input_fn(shape, dtype, device):
    dim = 0
    w_grad = torch.randn(shape, dtype=dtype, device=device)
    saved_v = torch.randn(shape, dtype=dtype, device=device)
    saved_g = torch.randn(shape[dim], dtype=dtype, device=device)
    saved_norms = torch.randn(shape[dim], dtype=dtype, device=device)
    yield w_grad, saved_v, saved_g, saved_norms, dim


@pytest.mark.weight_norm_interface_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface_backward():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="weight_norm_interface_backward",
        input_fn=weight_norm_interface_backward_input_fn,
        torch_op=torch.ops.aten._weight_norm_interface_backward,
        # NOTE: torch.ops.aten._weight_norm_interface_backward only supports float32,
        # using fp16/bf16 causes "expected scalar type Float but found Half" error.
        # Original: dtypes=consts.FLOAT_DTYPES,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.weight_norm_interface_backward)

    bench.run()
