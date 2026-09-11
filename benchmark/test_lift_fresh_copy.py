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


@pytest.mark.lift_fresh_copy
def test_lift_fresh_copy():
    bench = base.GenericBenchmark(
        input_fn=lambda shape, dtype, device: (
            iter([(torch.randn(shape, dtype=dtype, device=device),)])
        ),
        op_name="lift_fresh_copy",
        torch_op=torch.ops.aten.lift_fresh_copy,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
