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

import flag_gems

from . import base, consts
from .test_scatter_reduce import (
    LegacyScatterReduceTiming,
    scatter_reduce_input_fn_factory,
    scatter_reduce_more_shapes,
)


class ScatterReduceInplaceBenchmark(
    LegacyScatterReduceTiming, base.GenericBenchmark2DOnly
):
    def set_more_shapes(self):
        generic_shapes = super().set_more_shapes()
        scatter_shapes = scatter_reduce_more_shapes(generic_shapes)
        return list(dict.fromkeys(scatter_shapes + generic_shapes))


@pytest.mark.scatter_reduce_
@pytest.mark.skip_native(
    vendors="ascend",
    reason="Native aten::scatter_.reduce has no valid NPU kernel timing",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_scatter_reduce_legacy_inplace():
    bench = ScatterReduceInplaceBenchmark(
        op_name="scatter_reduce_",
        torch_op=torch.ops.aten.scatter_.reduce,
        gems_op=flag_gems.scatter_,
        input_fn=scatter_reduce_input_fn_factory(is_inplace=True, legacy=True),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
