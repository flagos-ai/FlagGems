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


def fill_scalar_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    yield input, 3.14159,


@pytest.mark.fill_scalar
def test_fill_scalar():
    bench = base.GenericBenchmark(
        op_name="fill_scalar",
        input_fn=fill_scalar_input_fn,
        torch_op=torch.fill,
        is_inplace=True,
    )
    bench.run()


def fill_tensor_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    yield input, 3.14159,


@pytest.mark.fill_tensor
def test_fill_tensor():
    bench = base.GenericBenchmark(
        op_name="fill_tensor",
        input_fn=fill_tensor_input_fn,
        torch_op=torch.fill,
        is_inplace=True,
    )
    bench.run()


def fill_inplace_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    yield input, 3.14159,


@pytest.mark.fill_tensor_
def test_fill_tensor_inplace():
    bench = base.GenericBenchmark(
        op_name="fill_tensor_",
        input_fn=fill_inplace_input_fn,
        torch_op=torch.fill_,
        is_inplace=True,
    )
    bench.run()


def fill_tensor_out_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    value = torch.tensor(3.14159, dtype=dtype, device=device)
    out = torch.empty_like(input)
    yield input, value, {"out": out}


@pytest.mark.fill_tensor_out
def test_fill_tensor_out():
    bench = base.GenericBenchmark(
        op_name="fill_tensor_out",
        input_fn=fill_tensor_out_input_fn,
        torch_op=torch.ops.aten.fill.Tensor_out,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.fill_scalar_
def test_fill_scalar_inplace():
    bench = base.GenericBenchmark(
        op_name="fill_scalar_",
        input_fn=fill_inplace_input_fn,
        torch_op=torch.ops.aten.fill_.Scalar,
        is_inplace=True,
    )
    bench.run()


def fill_scalar_out_input_fn(shape, dtype, device):
    input = torch.empty(shape, dtype=dtype, device=device)
    out = torch.empty_like(input)
    yield input, 3.14159, {"out": out}


@pytest.mark.fill_scalar_out
def test_fill_scalar_out():
    bench = base.GenericBenchmark(
        op_name="fill_scalar_out",
        input_fn=fill_scalar_out_input_fn,
        torch_op=torch.ops.aten.fill.Scalar_out,
        is_inplace=True,
    )
    bench.run()
