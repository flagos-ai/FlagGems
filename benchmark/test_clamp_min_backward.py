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

_CLAMP_MIN_BOUND = -0.5


def _clamp_min(inp):
    return torch.clamp_min(inp, _CLAMP_MIN_BOUND)


@pytest.mark.clamp_min_backward
def test_clamp_min_backward():
    # ``is_backward=True`` times ``torch.autograd.grad(out, xs, dout)``, so the
    # reference (torch.clamp_min's autograd) and the candidate
    # (flag_gems.clamp_min's ClampMinFunction) each run their own backward.
    bench = base.UnaryPointwiseBenchmark(
        op_name="clamp_min",
        torch_op=_clamp_min,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(functools.partial(flag_gems.clamp_min, mini=_CLAMP_MIN_BOUND))
    bench.run()
