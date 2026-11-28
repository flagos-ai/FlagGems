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

_CLAMP_MAX_BOUND = 0.5


def _clamp_max(inp):
    return torch.clamp_max(inp, _CLAMP_MAX_BOUND)


@pytest.mark.clamp_max_backward
def test_clamp_max_backward():
    # ``is_backward=True`` times ``torch.autograd.grad(out, xs, dout)``, so the
    # reference (torch.clamp_max's autograd) and the candidate
    # (flag_gems.clamp_max's ClampMaxFunction) each run their own backward.
    bench = base.UnaryPointwiseBenchmark(
        op_name="clamp_max",
        torch_op=_clamp_max,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(functools.partial(flag_gems.clamp_max, max_value=_CLAMP_MAX_BOUND))
    bench.run()
