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
from flag_gems.modules import GemsSiluAndMul
from flag_gems.testing import assert_close

from .module_test_util import has_vllm, init_seed

device = flag_gems.device


@pytest.mark.parametrize("shape", [(4, 64), (8, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gems_silu_and_mul(shape, dtype):
    init_seed(42)

    x1 = torch.randn(*shape, dtype=dtype, device=device)
    x2 = torch.randn(*shape, dtype=dtype, device=device)
    x_cat = torch.cat([x1, x2], dim=-1)

    target = GemsSiluAndMul()
    out_test = target(x1, x2)

    if has_vllm():
        from vllm.model_executor.layers.activation import SiluAndMul

        vmodule = SiluAndMul()
        vllm_ref = vmodule(x_cat)
        assert_close(out_test, vllm_ref, dtype, reduce_dim=shape[-1])

    else:
        pytest.skip("Skipping vLLM SiluAndMul comparison: vLLM not installed")
