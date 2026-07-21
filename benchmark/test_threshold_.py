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


def _input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp,


@pytest.mark.threshold_
def test_threshold_():
    bench = base.GenericBenchmark(
        op_name="threshold_",
        input_fn=_input_fn,
        torch_op=lambda x: torch.threshold_(x, 0.0, -1.0),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
