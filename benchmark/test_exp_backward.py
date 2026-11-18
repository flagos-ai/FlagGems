# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.ops.exp_backward import ExpFunction

from . import base, consts


class ExpBackwardBenchmark(base.UnaryPointwiseBenchmark):
    """``exp`` backward uses the default ``randn`` inputs.

    ``exp`` is smooth and finite over the whole real line, so the default
    ``torch.randn`` input distribution is representative; no domain clamping
    or custom sampler is needed.
    """


@pytest.mark.exp_backward
def test_exp_backward():
    # ``aten::exp`` keeps its own autograd formula node (exp recompute / mul),
    # so drive the gradient through the FlagGems autograd bridge explicitly --
    # otherwise this would time the native two-kernel chain, not this kernel.
    bench = ExpBackwardBenchmark(
        op_name="exp_backward",
        torch_op=torch.exp,
        gems_op=ExpFunction.apply,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
