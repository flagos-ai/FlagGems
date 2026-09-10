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


@pytest.mark.alias_copy
def test_alias_copy():
    bench = base.UnaryPointwiseBenchmark(
        op_name="alias_copy",
        torch_op=torch.ops.aten.alias_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.alias_copy_out
def test_alias_copy_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="alias_copy_out",
        torch_op=torch.ops.aten.alias_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
