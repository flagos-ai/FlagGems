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


@pytest.mark.cosh_backward
def test_cosh_backward():
    # ``is_backward=True`` makes the harness time
    # ``torch.autograd.grad(out, xs, dout)`` on the forward graph, so both the
    # reference (torch.cosh's autograd) and the candidate (flag_gems.cosh's
    # CoshFunction) exercise their own backward kernel.
    bench = base.UnaryPointwiseBenchmark(
        op_name="cosh",
        torch_op=torch.cosh,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(flag_gems.cosh)
    bench.run()
