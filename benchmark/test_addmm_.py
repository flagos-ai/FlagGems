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


@pytest.mark.addmm_
def test_addmm_(monkeypatch):
    def _input_fn(b, m, n, k, dtype, device, b_column_major):
        inp1 = torch.randn([m, k], dtype=dtype, device=device)
        bias = torch.randn([m, n], dtype=dtype, device=device)
        if b_column_major:
            inp2 = torch.randn([n, k], dtype=dtype, device=device)
            yield bias, inp1, inp2.t(),
        else:
            inp2 = torch.randn([k, n], dtype=dtype, device=device)
            yield bias, inp1, inp2,

    bench = base.BlasBenchmark(
        op_name="addmm_",
        input_fn=_input_fn,
        torch_op=lambda bias, mat1, mat2: bias.addmm_(mat1, mat2),
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
