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

from . import base, consts


class FloatPowerBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(4,), (1024,), (65536,), (512, 512), (2048, 2048)]


def _tensor_tensor_input(shape, dtype, device):
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    yield base_tensor, exponent


def _tensor_scalar_input(shape, dtype, device):
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    yield base_tensor, 1.234


def _scalar_tensor_input(shape, dtype, device):
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    yield 2.0, exponent


def _tensor_tensor_out_input(shape, dtype, device):
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    out = torch.empty(shape, dtype=torch.float64, device=device)
    yield base_tensor, exponent, {"out": out}


def _tensor_scalar_out_input(shape, dtype, device):
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    out = torch.empty(shape, dtype=torch.float64, device=device)
    yield base_tensor, 1.234, {"out": out}


def _scalar_tensor_out_input(shape, dtype, device):
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    out = torch.empty(shape, dtype=torch.float64, device=device)
    yield 2.0, exponent, {"out": out}


def _run_benchmark(op_name, input_fn, torch_op):
    bench = FloatPowerBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.float_power_tensor_tensor
def test_float_power_tensor_tensor():
    _run_benchmark(
        "float_power_tensor_tensor",
        _tensor_tensor_input,
        torch.ops.aten.float_power.Tensor_Tensor,
    )


@pytest.mark.float_power_tensor_scalar
def test_float_power_tensor_scalar():
    _run_benchmark(
        "float_power_tensor_scalar",
        _tensor_scalar_input,
        torch.ops.aten.float_power.Tensor_Scalar,
    )


@pytest.mark.float_power_scalar_tensor
def test_float_power_scalar_tensor():
    _run_benchmark(
        "float_power_scalar_tensor",
        _scalar_tensor_input,
        torch.ops.aten.float_power.Scalar,
    )


@pytest.mark.float_power_tensor_tensor_out
def test_float_power_tensor_tensor_out():
    _run_benchmark(
        "float_power_tensor_tensor_out",
        _tensor_tensor_out_input,
        torch.ops.aten.float_power.Tensor_Tensor_out,
    )


@pytest.mark.float_power_tensor_scalar_out
def test_float_power_tensor_scalar_out():
    _run_benchmark(
        "float_power_tensor_scalar_out",
        _tensor_scalar_out_input,
        torch.ops.aten.float_power.Tensor_Scalar_out,
    )


@pytest.mark.float_power_scalar_tensor_out
def test_float_power_scalar_tensor_out():
    _run_benchmark(
        "float_power_scalar_tensor_out",
        _scalar_tensor_out_input,
        torch.ops.aten.float_power.Scalar_out,
    )
