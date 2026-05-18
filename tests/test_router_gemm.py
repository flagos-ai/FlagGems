import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    ROUTER_SHAPES = [
        (16, 256, 7168),
    ]
    INPUT_DTYPES = [torch.bfloat16]
else:
    ROUTER_SHAPES = [
        (1, 256, 7168),
        (8, 256, 7168),
        (16, 256, 7168),
        (32, 256, 7168),
        (128, 256, 7168),
        (1, 64, 4096),
        (16, 128, 4096),
        (64, 384, 7168),
    ]
    INPUT_DTYPES = [torch.bfloat16]


@pytest.mark.router_gemm
@pytest.mark.parametrize("M, N, K", ROUTER_SHAPES)
@pytest.mark.parametrize("in_dtype", INPUT_DTYPES)
def test_router_gemm_accuracy(M, N, K, in_dtype):
    """Test that router_gemm (bf16 input -> fp32 output) matches fp32 reference."""
    x = torch.randn((M, K), dtype=in_dtype, device=flag_gems.device)
    weight = torch.randn((N, K), dtype=in_dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.mm(ref_x, ref_weight.t())

    with flag_gems.use_gems():
        res_out = flag_gems.router_gemm(x, weight)

    assert res_out.shape == (M, N)
    assert res_out.dtype == torch.float32
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=K)


@pytest.mark.router_gemm
@pytest.mark.parametrize("M, N, K", [(16, 256, 7168)])
def test_router_gemm_output_dtype(M, N, K):
    """Verify output dtype is fp32 (MoE router gate requirement)."""
    x = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    weight = torch.randn((N, K), dtype=torch.bfloat16, device=flag_gems.device)

    with flag_gems.use_gems():
        out_fp32 = flag_gems.router_gemm(x, weight)
        assert out_fp32.dtype == torch.float32
