import pytest
import torch

import flag_gems
from flag_gems.ops.awq_gemm import awq_gemm, pack_awq_weight

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    AWQ_SHAPES = [
        (1, 4096, 11008),  # decode (M = 1)
        (64, 512, 4096),  # prefill-like
    ]
    GROUP_SIZES = [128]
else:
    AWQ_SHAPES = [
        (1, 4096, 11008),  # decode (M = 1)
        (4, 4096, 11008),
        (32, 4096, 11008),
        (128, 256, 7168),
        (256, 512, 1024),
        (2048, 11008, 4096),
    ]
    GROUP_SIZES = [64, 128, 256]

INPUT_DTYPES = [torch.float16, torch.bfloat16]


def awq_reference(x, qweight, qzeros, scales, group_size):
    """Exact dequantized reference: out = input @ ((code - zero) * scale)."""
    K = x.shape[1]
    N = qweight.shape[1]
    G = K // group_size
    device = x.device
    w_code = torch.zeros(K, N, dtype=torch.int32, device=device)
    for i in range(8):
        w_code[i::8, :] = (qweight >> (4 * i)) & 0xF
    w_code = w_code.to(torch.float32)
    scales_f = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    if qzeros is not None:
        z_n = torch.zeros(G, N, dtype=torch.int32, device=device)
        for j in range(8):
            z_n[:, j::8] = (qzeros >> (4 * j)) & 0xF
        z_k = z_n.to(torch.float32).repeat_interleave(group_size, dim=0)
        w = (w_code - z_k) * scales_f
    else:
        w = w_code * scales_f
    return x.to(torch.float32) @ w


def make_awq_weights(K, N, group_size, in_dtype, has_zeros):
    w = torch.randn(K, N, device=flag_gems.device, dtype=in_dtype) * 0.1
    qweight, scales, qzeros = pack_awq_weight(w, group_size, dtype=in_dtype)
    if not has_zeros:
        qzeros = None
    return qweight, qzeros, scales


@pytest.mark.awq_gemm
@pytest.mark.parametrize("M, N, K", AWQ_SHAPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("dtype", INPUT_DTYPES)
@pytest.mark.parametrize("has_zeros", [True, False])
def test_awq_gemm_accuracy(M, N, K, group_size, dtype, has_zeros):
    """awq_gemm matches the exact dequantized matmul reference."""
    x = torch.randn(M, K, device=flag_gems.device, dtype=dtype)
    qweight, qzeros, scales = make_awq_weights(K, N, group_size, dtype, has_zeros)

    ref_out = awq_reference(x, qweight, qzeros, scales, group_size)
    res_out = awq_gemm(x, qweight, qzeros, scales, group_size)

    assert res_out.shape == (M, N)
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.awq_gemm
@pytest.mark.parametrize("M, N, K", [(64, 512, 4096)])
@pytest.mark.parametrize("group_size", [128])
def test_awq_gemm_fp32_input(M, N, K, group_size):
    """fp32 input with fp32 output is supported."""
    x = torch.randn(M, K, device=flag_gems.device, dtype=torch.float32)
    qweight, qzeros, scales = make_awq_weights(
        K, N, group_size, torch.float16, has_zeros=True
    )

    ref_out = awq_reference(x, qweight, qzeros, scales, group_size)
    res_out = awq_gemm(x, qweight, qzeros, scales, group_size, out_dtype=torch.float32)

    assert res_out.dtype == torch.float32
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=K)


@pytest.mark.awq_gemm
@pytest.mark.parametrize("M, N, K", [(16, 512, 4096)])
@pytest.mark.parametrize("group_size", [128])
def test_awq_gemm_out_dtype(M, N, K, group_size):
    """out_dtype can override the output dtype."""
    x = torch.randn(M, K, device=flag_gems.device, dtype=torch.float16)
    qweight, qzeros, scales = make_awq_weights(
        K, N, group_size, torch.float16, has_zeros=True
    )
    res = awq_gemm(x, qweight, qzeros, scales, group_size, out_dtype=torch.float32)
    assert res.dtype == torch.float32
