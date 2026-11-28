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

import flag_gems

from . import base, consts


def dist_backward_input_fn(shape, dtype, device):
    # shape is a 1-D plan; the backward of the global p-norm runs over the same
    # flat layout as the forward, so both inputs share the shape and the
    # upstream gradient is a scalar.
    inp = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn((), dtype=dtype, device=device)
    yield grad, inp, other


def _dist_backward_ref(grad, inp, other):
    """Eager reference with the same signature as the FlagGems wrapper.

    ``DistBackward0`` evaluates ``norm_backward(grad, self - other, p, result)``;
    the p = 2 branch of the Triton kernel is the same signed quotient.
    """
    dist = torch.dist(inp, other, 2).detach()
    return grad * (inp - other) / dist, -grad * (inp - other) / dist


class DistBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Flat 1-D plans: the backward reads the whole difference and writes
        # both gradients once, so latency scales with element count.
        self.shapes = [
            (1024,),
            (4096,),
            (16384,),
            (65536,),
        ]

    def set_more_shapes(self):
        return []


@pytest.mark.dist_backward
def test_dist_backward():
    """Backward of ``dist``: the signed p-norm derivative at both inputs.

    The Triton kernel computes both gradients in one launch; the eager
    reference walks the ATen autograd chain (``DistBackward0``).
    """
    bench = DistBackwardBenchmark(
        op_name="dist_backward",
        input_fn=dist_backward_input_fn,
        torch_op=_dist_backward_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.dist_backward)
    bench.run()
