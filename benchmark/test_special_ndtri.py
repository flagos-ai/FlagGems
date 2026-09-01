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


@pytest.mark.special_ndtri
def test_special_ndtri():
    def special_ndtri_input_fn(shape, dtype, device):
        # ndtri takes a probability in [0, 1]. The default randn input would be
        # mostly out of range and would degenerate into nan, so the domain is
        # sampled explicitly here.
        yield torch.empty(shape, dtype=dtype, device=device).uniform_(0.01, 0.99),

    bench = base.GenericBenchmark(
        input_fn=special_ndtri_input_fn,
        op_name="special_ndtri",
        torch_op=torch.ops.aten.special_ndtri,
        # ndtri is implemented for single and double precision only, so the
        # half precision dtypes benchmarked for most pointwise ops are excluded.
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
