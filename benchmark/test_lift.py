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

# aten::lift is a metadata-only op: it returns the input tensor unchanged and
# never launches a device kernel, so --kernel-check has nothing to profile for
# it. The lift.out benchmark below does launch a copy kernel and is checked.
_KERNEL_CHECK = getattr(base.Config, "kernel_check", False)


@pytest.mark.lift
@pytest.mark.skipif(_KERNEL_CHECK, reason="aten::lift launches no device kernel")
def test_lift():
    bench = base.UnaryPointwiseBenchmark(
        op_name="lift",
        torch_op=torch.ops.aten.lift,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.lift_out
def test_lift_out():
    # Uses UnaryPointwiseOutBenchmark (passes out= via get_input_iter)
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="lift_out",
        torch_op=torch.ops.aten.lift.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
