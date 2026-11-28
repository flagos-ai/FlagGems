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

from typing import Generator

import pytest
import torch

from . import base


def _ref(inp, scale, zero_point):
    return torch.ops.aten._make_per_tensor_quantized_tensor(inp, scale, zero_point)


def _out_ref(inp, scale, zero_point, out):
    return torch.ops.aten._make_per_tensor_quantized_tensor.out(
        inp, scale, zero_point, out=out
    )


def _int_input(shape, dtype, device):
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=dtype, device=device)
    return torch.randint(-128, 128, shape, dtype=dtype, device=device)


class MakePerTensorQuantizedTensorBenchmark(base.UnaryPointwiseBenchmark):
    """Unary integer copy plus the constant ``scale``/``zero_point`` tags."""

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = _int_input(shape, cur_dtype, self.device)
            yield inp, 0.25, 7


class MakePerTensorQuantizedTensorOutBenchmark(base.UnaryPointwiseOutBenchmark):
    """``.out`` variant: also passes the constant ``scale``/``zero_point``."""

    def get_input_iter(self, cur_dtype) -> Generator:
        qdtype = {
            torch.uint8: torch.quint8,
            torch.int8: torch.qint8,
            torch.int32: torch.qint32,
        }[cur_dtype]
        for shape in self.shapes:
            inp = _int_input(shape, cur_dtype, self.device)
            out = torch._empty_affine_quantized(
                shape, dtype=qdtype, device=self.device, scale=0.25, zero_point=7
            )
            yield inp, 0.25, 7, {"out": out}


@pytest.mark.make_per_tensor_quantized_tensor
def test_make_per_tensor_quantized_tensor():
    bench = MakePerTensorQuantizedTensorBenchmark(
        op_name="make_per_tensor_quantized_tensor",
        torch_op=_ref,
        dtypes=[torch.uint8],
    )
    bench.run()


@pytest.mark.make_per_tensor_quantized_tensor_out
def test_make_per_tensor_quantized_tensor_out():
    bench = MakePerTensorQuantizedTensorOutBenchmark(
        op_name="make_per_tensor_quantized_tensor_out",
        torch_op=_out_ref,
        dtypes=[torch.uint8],
    )
    bench.run()
