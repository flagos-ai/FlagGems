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

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    CONJ_SHAPES = [(32, 64)]
    CONJ_DTYPES = [torch.float32]
else:
    CONJ_SHAPES = [(256,), (32, 64), (2, 3, 4)]
    CONJ_DTYPES = [torch.float16, torch.float32, torch.bfloat16]


@pytest.mark.conj_physical
@pytest.mark.parametrize("shape", CONJ_SHAPES)
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("dtype", CONJ_DTYPES)
def test_conj_physical(shape, is_complex, dtype):
    device = flag_gems.device

    if is_complex:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        input = torch.complex(real, imag)
        out_dtype = input.dtype
    else:
        input = torch.randn(shape, dtype=dtype, device=device)
        out_dtype = dtype

    ref_input = utils.to_reference(input, True)
    ref_out = torch.conj_physical(ref_input)
    with flag_gems.use_gems():
        res_out = torch.conj_physical(input)

    utils.gems_assert_close(res_out, ref_out, out_dtype, reduce_dim=1)
