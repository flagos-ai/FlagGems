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

from . import base, consts


def _atleast_3d_input_fn(shape, cur_dtype, device):
    yield base.generate_tensor_input(shape, cur_dtype, device),


@pytest.mark.atleast_3d
def test_atleast_3d():
    bench = base.GenericBenchmark(
        op_name="atleast_3d",
        input_fn=_atleast_3d_input_fn,
        torch_op=torch.atleast_3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
