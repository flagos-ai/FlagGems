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


@pytest.mark.not_equal_
def test_not_equal_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="not_equal_",
        torch_op=lambda a, b: torch.ops.aten.ne_.Tensor(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def not_equal_scalar_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp, 0.5


@pytest.mark.not_equal_scalar_
def test_not_equal_scalar_():
    bench = base.GenericBenchmark(
        op_name="not_equal_scalar_",
        input_fn=not_equal_scalar_input_fn,
        torch_op=lambda a, b: torch.ops.aten.ne_.Scalar(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
