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

# quantize_per_tensor_dynamic takes a float32 input and a target quantized
# dtype (torch.quint8 / torch.qint8) plus a reduce_range flag. We use the
# quantized dtypes as the benchmark "dtypes" and always feed a float32 input.
QUANT_DTYPES = [torch.quint8, torch.qint8]


def _quant_input_fn(shape, qdtype, device):
    # The framework passes the current "dtype" (here a quantized dtype) through;
    # the actual input tensor is float32. Benchmark both reduce_range values.
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    for reduce_range in (False, True):
        yield inp, qdtype, reduce_range


@pytest.mark.quantize_per_tensor_dynamic
def test_quantize_per_tensor_dynamic():
    bench = base.GenericBenchmark(
        op_name="quantize_per_tensor_dynamic",
        torch_op=torch.quantize_per_tensor_dynamic,
        input_fn=_quant_input_fn,
        dtypes=QUANT_DTYPES,
    )
    bench.run()


@pytest.mark.quantize_per_tensor_dynamic_out
def test_quantize_per_tensor_dynamic_out():
    def out_input_fn(shape, qdtype, device):
        inp = torch.randn(shape, dtype=torch.float32, device=device)
        for reduce_range in (False, True):
            # Pre-build a quantized out tensor; its scale/zp get overwritten.
            out = torch.quantize_per_tensor(
                torch.zeros(shape, device=device),
                scale=1.0,
                zero_point=0,
                dtype=qdtype,
            )
            yield inp, qdtype, reduce_range, {"out": out}

    def op_fn(inp, qdtype, reduce_range, *, out=None):
        return torch.ops.aten.quantize_per_tensor_dynamic.out(
            inp, qdtype, reduce_range, out=out
        )

    bench = base.GenericBenchmark(
        op_name="quantize_per_tensor_dynamic_out",
        torch_op=op_fn,
        input_fn=out_input_fn,
        dtypes=QUANT_DTYPES,
    )
    bench.run()
