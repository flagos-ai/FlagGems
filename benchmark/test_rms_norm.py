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


@pytest.mark.rms_norm
def test_rms_norm():
    def rms_norm_input_fn(shape, dtype, device):
        _, N = shape
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        yield inp, (N,), weight

    bench = base.GenericBenchmark2DOnly(
        op_name="rms_norm",
        input_fn=rms_norm_input_fn,
        torch_op=torch.nn.functional.rms_norm,
    )
    bench.run()
