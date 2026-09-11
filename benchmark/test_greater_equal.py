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


def _greater_equal_scalar_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, 0.5


@pytest.mark.greater_equal
def test_greater_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater_equal",
        torch_op=torch.greater_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.greater_equal_
def test_greater_equal_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="greater_equal_",
        torch_op=lambda a, b: a.greater_equal_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.greater_equal_scalar
def test_greater_equal_scalar():
    bench = base.GenericBenchmark(
        input_fn=_greater_equal_scalar_input_fn,
        op_name="greater_equal_scalar",
        torch_op=torch.greater_equal,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
