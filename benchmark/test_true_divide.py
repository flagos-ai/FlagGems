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


def _true_divide_scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.5


@pytest.mark.true_divide_tensor
def test_true_divide_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="true_divide_tensor",
        torch_op=torch.ops.aten.true_divide.Tensor,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.true_divide_scalar
def test_true_divide_scalar():
    bench = base.GenericBenchmark(
        input_fn=_true_divide_scalar_input_fn,
        op_name="true_divide_scalar",
        torch_op=torch.ops.aten.true_divide.Scalar,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.true_divide__scalar
def test_true_divide__scalar():
    bench = base.GenericBenchmark(
        input_fn=_true_divide_scalar_input_fn,
        op_name="true_divide__scalar",
        torch_op=torch.ops.aten.true_divide_.Scalar,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.true_divide_out
def test_true_divide_out():
    def input_fn(shape, dtype, device):
        inp1 = utils.generate_tensor_input(shape, dtype, device)
        inp2 = utils.generate_tensor_input(shape, dtype, device)
        out = torch.empty_like(inp1)
        yield inp1, inp2, {"out": out}

    bench = base.GenericBenchmark(
        op_name="true_divide_out",
        input_fn=input_fn,
        torch_op=torch.ops.aten.true_divide.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.true_divide__tensor
def test_true_divide__tensor_dispatch():
    bench = base.BinaryPointwiseBenchmark(
        op_name="true_divide__tensor",
        torch_op=torch.ops.aten.true_divide_.Tensor,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
