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

from . import base

# ``quantize_per_tensor`` only accepts float32 tensors, so the benchmark input is
# always float32; the produced quantized tensor is ``torch.quint8`` (the most
# common per-tensor quantization dtype). ``scale``/``zero_point`` are fixed so
# the comparison against the Torch reference is apples-to-apples.
SCALE = 0.1
ZERO_POINT = 10
Q_DTYPE = torch.quint8


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    yield inp, SCALE, ZERO_POINT, Q_DTYPE


def _input_fn_out(shape, dtype, device):
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    out = torch.quantize_per_tensor(inp, 0.5, 100, Q_DTYPE)
    yield inp, SCALE, ZERO_POINT, Q_DTYPE, {"out": out}


@pytest.mark.quantize_per_tensor
def test_quantize_per_tensor():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="quantize_per_tensor",
        input_fn=_input_fn,
        # quantize_per_tensor only accepts float32 on the quantized CUDA backend.
        dtypes=[torch.float32],
        torch_op=torch.quantize_per_tensor,
    )
    bench.run()


@pytest.mark.quantize_per_tensor_out
def test_quantize_per_tensor_out():
    def torch_op(a, scale, zero_point, dtype, out=None):
        return torch.ops.aten.quantize_per_tensor.out(
            a, scale, zero_point, dtype, out=out
        )

    bench = base.GenericBenchmarkExcluse1D(
        op_name="quantize_per_tensor_out",
        input_fn=_input_fn_out,
        # quantize_per_tensor only accepts float32 on the quantized CUDA backend.
        dtypes=[torch.float32],
        torch_op=torch_op,
    )
    bench.run()
