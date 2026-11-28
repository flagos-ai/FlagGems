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
"""Benchmark for the gradient of ``clone``.

Both sides run the same autograd chain -- a forward ``clone`` that builds the
graph, then ``torch.autograd.grad`` for the backward -- so the comparison is
like-for-like. Timing the FlagGems backward helper against a straight
``CopyBackwards`` pass-through would not be: the reference forwards
``grad_output`` with no device work at all (see
``tests/test_clone_backward.py::test_clone_backward_is_an_identity_pass_through``),
so the "speedup" would only be measuring a copy against a no-op.
"""

import pytest
import torch

import flag_gems

from . import base, consts

# bfloat16 has no kernel on every backend; this is a static capability query, so
# case listing stays free of device work and tensor allocations.
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


@pytest.mark.clone_backward
def test_clone_backward():
    # ``is_backward=True`` makes the harness time
    # ``torch.autograd.grad(out, xs, dout)`` on the forward graph, so the
    # reference (torch.clone's CopyBackwards) and the candidate
    # (flag_gems.clone's _CloneFunction) each exercise their own backward.
    bench = base.UnaryPointwiseBenchmark(
        op_name="clone",
        torch_op=torch.clone,
        dtypes=_BENCH_DTYPES,
        is_backward=True,
    )
    bench.set_gems(flag_gems.clone)
    bench.run()
