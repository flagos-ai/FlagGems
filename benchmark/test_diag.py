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

import random

import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    input = utils.generate_tensor_input(shape, dtype, device)
    diagonal = random.randint(-4, 4)
    yield input, {
        "diagonal": diagonal,
    },


@pytest.mark.diag
def test_diag():
    bench = base.GenericBenchmarkExcluse3D(
        op_name="diag",
        input_fn=_input_fn,
        torch_op=torch.diag,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )

    bench.run()
