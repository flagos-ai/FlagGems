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


@pytest.mark.divide_tensor
def test_divide_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="divide_tensor",
        torch_op=torch.ops.aten.divide.Tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.divide__tensor
def test_divide__tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="divide__tensor",
        torch_op=torch.ops.aten.divide_.Tensor,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _divide_scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.5


@pytest.mark.divide_scalar
def test_divide_scalar():
    bench = base.GenericBenchmark(
        op_name="divide_scalar",
        input_fn=_divide_scalar_input_fn,
        torch_op=torch.ops.aten.divide.Scalar,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.divide__scalar
def test_divide__scalar():
    bench = base.GenericBenchmark(
        op_name="divide__scalar",
        input_fn=_divide_scalar_input_fn,
        torch_op=torch.ops.aten.divide_.Scalar,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _divide_scalar_mode_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, -2.5


def _divide_tensor_mode_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    other = utils.generate_tensor_input(shape, dtype, device)
    other = torch.where(other >= 0, other + 0.1, other - 0.1)
    yield inp, other


@pytest.mark.divide_scalar_mode
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_divide_scalar_mode(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="divide_scalar_mode",
        input_fn=_divide_scalar_mode_input_fn,
        torch_op=lambda a, b: torch.ops.aten.divide.Scalar_mode(
            a, b, rounding_mode=rounding_mode
        ),
        dtypes=[torch.float32] if rounding_mode == "trunc" else consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.divide__scalar_mode
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_divide__scalar_mode(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="divide__scalar_mode",
        input_fn=_divide_scalar_mode_input_fn,
        torch_op=lambda a, b: torch.ops.aten.divide_.Scalar_mode(
            a, b, rounding_mode=rounding_mode
        ),
        dtypes=[torch.float32] if rounding_mode == "trunc" else consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.divide_tensor_mode
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_divide_tensor_mode(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="divide_tensor_mode",
        input_fn=_divide_tensor_mode_input_fn,
        torch_op=lambda a, b: torch.ops.aten.divide.Tensor_mode(
            a, b, rounding_mode=rounding_mode
        ),
        dtypes=[torch.float32] if rounding_mode == "trunc" else consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.divide__tensor_mode
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_divide__tensor_mode(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="divide__tensor_mode",
        input_fn=_divide_tensor_mode_input_fn,
        torch_op=lambda a, b: torch.ops.aten.divide_.Tensor_mode(
            a, b, rounding_mode=rounding_mode
        ),
        dtypes=[torch.float32] if rounding_mode == "trunc" else consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
