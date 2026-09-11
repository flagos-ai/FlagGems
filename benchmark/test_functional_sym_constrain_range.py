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


def _functional_sym_constrain_range_input_fn(shape, cur_dtype, device):
    dep_token = base.generate_tensor_input(shape, cur_dtype, device)
    yield 5, 1, 10, dep_token


@pytest.mark.functional_sym_constrain_range
def test_functional_sym_constrain_range():
    bench = base.GenericBenchmark(
        op_name="functional_sym_constrain_range",
        torch_op=torch.ops.aten._functional_sym_constrain_range,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=_functional_sym_constrain_range_input_fn,
    )
    bench.run()
