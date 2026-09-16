import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# FP16/BF16 only: int8 matmul requires half-precision activation
WEIGHT_INT8PACK_MM_DTYPES = [torch.float16, torch.bfloat16]


# M: number of tokens / rows in activation
# N: output features (weight row count)
# K: input features (weight column count)
WEIGHT_INT8PACK_MM_SHAPES = [
    (16, 128, 256),
    (16, 128, 512),
    (16, 256, 256),
    (16, 256, 512),
    (64, 128, 256),
    (64, 128, 512),
    (64, 256, 256),
    (64, 256, 512),
]


@pytest.mark.weight_int8pack_mm
@pytest.mark.parametrize("M, N, K", WEIGHT_INT8PACK_MM_SHAPES)
@pytest.mark.parametrize("dtype", WEIGHT_INT8PACK_MM_DTYPES)
def test_weight_int8pack_mm(M, N, K, dtype):
    A = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    # B is int8 weight, shape (N, K) — one byte per element (not bit-packed)
    B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=flag_gems.device)
    scales = torch.randn((N,), dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A, False)
    ref_B = utils.to_reference(B, False)
    ref_scales = utils.to_reference(scales, False)

    with flag_gems.use_gems():
        res_out = flag_gems.weight_int8pack_mm(A, B, scales)

    ref_out = _ref_weight_int8pack_mm(ref_A, ref_B, ref_scales)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


def _ref_weight_int8pack_mm(A, B, scales):
    """Reference implementation using PyTorch eager mode."""
    B_fp = B.to(A.dtype)
    result = torch.matmul(A, B_fp.T)
    result = result * scales.unsqueeze(0)
    return result
