import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.config import has_c_extension
from flag_gems.modules import GemsRMSNorm
from flag_gems.testing import assert_close

from .module_test_util import has_vllm, init_seed, is_torch_version_ge

device = flag_gems.device


# TODO(flaggems): Current implementation only supports 2D input tensors and a 1D `normalized_shape` (given as int).
# Need to extend support for multi-dimensional `normalized_shape` and multi-dimensional input tensors.,
# as supported by `torch.nn.RMSNorm`.
@pytest.mark.parametrize("shape", [(4, 64), (8, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gems_rmsnorm(shape, dtype):
    init_seed(42)
    norm_shape = shape[-1]
    np_inp = np.random.uniform(-0.1, 0.1, shape).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, norm_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=device)
    weight = torch.tensor(np_weight, dtype=dtype, device=device)

    target = GemsRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    target.weight.data.copy_(weight.data)
    out_test = target(inp)

    if is_torch_version_ge("2.4.0"):
        from torch.nn import RMSNorm

        reference = RMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
        reference.weight.data.copy_(weight)
        torch_ref = reference(inp)

        assert_close(out_test, torch_ref, dtype, reduce_dim=norm_shape)
    else:
        pytest.skip("Skipping PyTorch RMSNorm comparison: torch<2.4.0")

    if has_vllm():
        from vllm.model_executor.layers.layernorm import RMSNorm as VRMSNorm

        vllm_reference = VRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
        vllm_reference.weight.data.copy_(weight)
        vllm_ref = vllm_reference(inp)

        assert_close(out_test, vllm_ref, dtype, reduce_dim=norm_shape)
    else:
        pytest.skip("Skipping vLLM RMSNorm comparison: vLLM not installed")

    if has_c_extension:
        torch.library.opcheck(
            torch.ops.flag_gems.rms_norm,
            (inp, target.weight.data, target.eps),
            test_utils=("test_schema", "test_autograd_registration"),
        )


@pytest.mark.parametrize("shape", [(4, 64), (8, 128), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gems_rmsnorm_with_residual(shape, dtype):
    norm_shape = shape[-1]
    scale = 1 / (2 * norm_shape)
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, norm_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=device)
    weight = torch.tensor(np_weight, dtype=dtype, device=device)
    residual = torch.randn_like(inp) * scale

    target = GemsRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
    target.weight.data.copy_(weight)

    if is_torch_version_ge("2.4.0"):
        from torch.nn import RMSNorm

        reference = RMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
        reference.weight.data.copy_(weight)
        torch_inp = inp.clone()
        torch_residual = residual.clone()
        torch_new_residual = torch_inp + torch_residual
        torch_ref = reference(torch_new_residual)

        gems_inp = inp.clone()
        gems_residual = residual.clone()
        out_test, new_residual = target(gems_inp, gems_residual)

        assert_close(out_test, torch_ref, dtype, reduce_dim=norm_shape)
        assert_close(new_residual, torch_new_residual, dtype)
    else:
        pytest.skip("Skipping PyTorch RMSNorm comparison: torch<2.4.0")

    if has_vllm():
        from vllm.model_executor.layers.layernorm import RMSNorm as VRMSNorm

        vllm_reference = VRMSNorm(norm_shape, eps=1e-6).to(dtype=dtype, device=device)
        vllm_reference.weight.data.copy_(weight)
        vllm_inp = inp.clone()
        vllm_residual = residual.clone()
        vllm_ref, vllm_new_residual = vllm_reference(vllm_inp, vllm_residual)

        out_test, new_residual = target(inp.clone(), residual.clone())
        assert_close(out_test, vllm_ref, dtype, reduce_dim=norm_shape)
        assert_close(new_residual, vllm_new_residual, dtype)
    else:
        pytest.skip("Skipping vLLM RMSNorm comparison: vLLM not installed")

    if has_c_extension:
        torch.library.opcheck(
            torch.ops.flag_gems.fused_add_rms_norm,
            (inp, residual, target.weight.data, target.eps),
            test_utils=("test_schema", "test_autograd_registration"),
        )
