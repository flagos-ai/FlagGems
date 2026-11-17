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


def _cumsum_backward_ref(grad, input, dim):
    """Eager baseline with the same signature as ``flag_gems.cumsum_backward``.

    ``aten::cumsum`` has no dedicated backward operator, so autograd derives it
    as a flip + ``cumsum`` + flip chain. Timing that chain is the fair eager
    comparison for the single-kernel reverse scan. ``input`` only carries the
    shape/dtype/device, exactly as in the FlagGems wrapper.
    """
    return torch.flip(torch.cumsum(torch.flip(grad, [dim]), dim), [dim])


def cumsum_backward_input_fn(shape, dtype, device):
    # ``grad`` is the upstream gradient; the reverse scan reads it once and
    # writes once, so no forward output is needed (unlike ``cumprod_backward``).
    grad = utils.generate_tensor_input(shape, dtype, device)
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield grad, inp, 1


@pytest.mark.cumsum_backward
def test_cumsum_backward():
    # ``flag_gems.cumsum_backward`` is the explicit Triton entry: the
    # ``torch.*`` path would keep ATen's ``CumsumBackward0`` node. Drive the
    # FlagGems wrapper through ``set_gems`` so the reverse-scan kernel is what
    # actually gets timed.
    bench = base.GenericBenchmark2DOnly(
        op_name="cumsum_backward",
        input_fn=cumsum_backward_input_fn,
        torch_op=_cumsum_backward_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.cumsum_backward)
    bench.run()
