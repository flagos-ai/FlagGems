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


def cumsum_backward_input_fn(shape, dtype, device):
    # ``is_backward=True`` clones the tensor inputs with ``requires_grad`` and
    # times ``torch.autograd.grad`` on the forward output. The upstream gradient
    # is drawn bounded so the fp16 baseline stays finite.
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 1


@pytest.mark.cumsum_backward
def test_cumsum_backward():
    # ``aten::cumsum`` has no dedicated backward op, so under ``use_gems()`` the
    # ATen ``CumsumBackward0`` node stays above the device kernel and the
    # reference backward would win on the ``torch.*`` path. Drive the explicit
    # FlagGems entry through ``set_gems`` so the Triton reverse-scan kernel is
    # what actually gets timed.
    bench = base.GenericBenchmark2DOnly(
        op_name="cumsum_backward",
        input_fn=cumsum_backward_input_fn,
        torch_op=torch.cumsum,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.set_gems(flag_gems.cumsum)
    bench.run()
