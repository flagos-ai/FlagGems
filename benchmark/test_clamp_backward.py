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

import functools

import pytest
import torch

import flag_gems

from . import base, consts

_CLAMP_MIN = -0.5
_CLAMP_MAX = 0.5


def _clamp(inp):
    return torch.clamp(inp, min=_CLAMP_MIN, max=_CLAMP_MAX)


@pytest.mark.clamp_backward
def test_clamp_backward():
    # ``is_backward=True`` times ``torch.autograd.grad(out, xs, dout)``, so the
    # reference (torch.clamp's autograd) and the candidate (flag_gems.clamp's
    # ClampFunction) each run their own backward.
    bench = base.UnaryPointwiseBenchmark(
        op_name="clamp",
        torch_op=_clamp,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(functools.partial(flag_gems.clamp, mini=_CLAMP_MIN, maxi=_CLAMP_MAX))
    bench.run()
