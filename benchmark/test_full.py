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


def _input_fn(shape, dtype, device):
    yield {"size": shape, "fill_value": 3.1415926, "dtype": dtype, "device": device},


@pytest.mark.full
def test_full():
    bench = base.GenericBenchmark(
        op_name="full", input_fn=_input_fn, torch_op=torch.full
    )
    bench.run()
