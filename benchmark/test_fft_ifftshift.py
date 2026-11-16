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

from . import base, consts


def _fft_ifftshift_input_fn(shape, cur_dtype, device):
    # fft_ifftshift is shape-preserving, so only the dim to shift has to be
    # provided alongside the input tensor.
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    yield inp, {"dim": [-1]}


@pytest.mark.fft_ifftshift
def test_fft_ifftshift():
    bench = base.GenericBenchmark(
        op_name="fft_ifftshift",
        input_fn=_fft_ifftshift_input_fn,
        torch_op=torch.ops.aten.fft_ifftshift,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
