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

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference


@pytest.mark.native_norm
@pytest.mark.parametrize(
    "shape", [(1024,), (64, 64), (256, 512), (4, 256, 3), (32, 32, 32)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2, 3])
def test_native_norm(shape, dtype, p):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(inp, True)

    ref_out = torch.linalg.vector_norm(ref_inp.flatten(), float(p))
    with flag_gems.use_gems():
        res_out = torch.native_norm(inp, p)

    gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
