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

from . import base, consts, utils

KEEP_PROBABILITIES = [0.3, 0.5, 0.8]


def fused_dropout_backward_input_fn(shape, dtype, device):
    grad = utils.generate_tensor_input(shape, dtype, device)
    mask = (torch.rand(shape, device=device) < 0.5).to(torch.uint8)
    yield grad, mask, 0.5


def _fused_dropout_backward_ref(grad, mask, p):
    """Eager formula with the same signature as the FlagGems entry.

    ``aten::_fused_dropout`` has no CPU/CUDA ``_fused_dropout_backward``
    dispatch schema, so autograd materializes the derivative through
    ``_masked_scale``; timing the equivalent elementwise chain is the fair
    eager comparison for the single-kernel backward.
    """
    return grad * mask.to(grad.dtype) / p


@pytest.mark.fused_dropout_backward
def test_fused_dropout_backward():
    bench = base.GenericBenchmark2DOnly(
        op_name="fused_dropout_backward",
        input_fn=fused_dropout_backward_input_fn,
        torch_op=_fused_dropout_backward_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.fused_dropout_backward)
    bench.run()
