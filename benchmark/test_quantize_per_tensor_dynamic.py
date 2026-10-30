# Copyright 2026, The FlagOS Contributors.
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

# quantize_per_tensor_dynamic maps a float32 tensor onto a dynamically
# computed qint8/quint8 grid. The input dtype is always float32 (the operator
# rejects everything else on the quantized CUDA backend); the benchmark runs
# the qint8 scheme, whose torch.float32 entry is the input dtype that feed
# dtype parametrization requires.
QPTD_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
]


def _qptd_input_fn(shape, dtype, device):
    """Yield (input, qdtype, reduce_range) for the dynamic quantization."""
    x = torch.randn(shape, dtype=torch.float32, device=device) * 3.0
    yield x, torch.qint8, False


class QuantizePerTensorDynamicBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = QPTD_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark.quantize_per_tensor_dynamic
def test_quantize_per_tensor_dynamic():
    bench = QuantizePerTensorDynamicBenchmark(
        op_name="quantize_per_tensor_dynamic",
        torch_op=torch.quantize_per_tensor_dynamic,
        input_fn=_qptd_input_fn,
        # torch.float32 is the only accepted input dtype on the quantized CUDA
        # backend (others raise "Quantize only works on Float Tensor"), so the
        # list cannot be widened to consts.FLOAT_DTYPES.
        dtypes=[torch.float32],
    )
    bench.run()


def _qptd_out_input_fn(shape, dtype, device):
    """Yield (input, qdtype, reduce_range, {'out': out})."""
    x = torch.randn(shape, dtype=torch.float32, device=device) * 3.0
    # Pre-allocate the per-tensor quantized ``out`` through the public
    # quantize_per_tensor factory; the FlagGems out kernel overwrites its
    # integer storage and adopts the dynamically computed quantizer
    # (quantized copy_ semantics).
    out = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=device),
        0.9,
        3,
        torch.qint8,
    )
    yield x, torch.qint8, False, {"out": out}


class QuantizePerTensorDynamicOutBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = QPTD_SHAPES

    def set_more_shapes(self):
        return []


@pytest.mark.quantize_per_tensor_dynamic_out
def test_quantize_per_tensor_dynamic_out():
    bench = QuantizePerTensorDynamicOutBenchmark(
        op_name="quantize_per_tensor_dynamic_out",
        torch_op=torch.ops.aten.quantize_per_tensor_dynamic.out,
        input_fn=_qptd_out_input_fn,
        # torch.float32 is the only accepted input dtype on the quantized CUDA
        # backend (others raise "Quantize only works on Float Tensor"), so the
        # list cannot be widened to consts.FLOAT_DTYPES.
        dtypes=[torch.float32],
    )
    bench.run()
