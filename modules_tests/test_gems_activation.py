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
