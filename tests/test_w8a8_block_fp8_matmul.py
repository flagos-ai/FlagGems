import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

FP8_MNK_SHAPES = [
    (128, 256, 512),
    (64, 128, 128),
    (256, 256, 256),
    (83, 7748, 3884),
    (84, 7168, 3884),
]


@pytest.mark.w8a8_block_fp8_matmul
@pytest.mark.parametrize("M,N,K", FP8_MNK_SHAPES)
def test_w8a8_block_fp8_matmul(M, N, K):
    if flag_gems.vendor_name == "mthreads":
        if hasattr(torch, "float8_e4m3fn"):
            dtype = torch.float8_e4m3fn
        else:
            dtype = torch.float32
    else:
        if not torch.cuda.is_available():
            pytest.skip("w8a8_block_fp8_matmul test requires CUDA or mthreads")
        major, _ = torch.cuda.get_device_capability()
        if major > 8:
            dtype = torch.float8_e4m3fn
        elif major == 8:
            dtype = torch.float8_e5m2
        else:
            dtype = torch.float32

    device = flag_gems.device
    block_n = 128
    block_k = 128
    block_size = [block_n, block_k]

    A = torch.randn((M, K), device=device).to(dtype)
    B = torch.randn((N, K), device=device).to(dtype)

    num_k_groups = (K + block_k - 1) // block_k
    num_n_groups = (N + block_n - 1) // block_n

    As = (0.01 * torch.rand(M, num_k_groups, device=device) + 0.005).to(torch.float32)
    Bs = (0.01 * torch.rand(num_n_groups, num_k_groups, device=device) + 0.005).to(
        torch.float32
    )

    A_ref = A.to(torch.float32)
    B_ref = B.to(torch.float32)
    As_ref = As.to(torch.float32)
    Bs_ref = Bs.to(torch.float32)

    A_scaled = torch.zeros_like(A_ref)
    for k_group in range(num_k_groups):
        k_start = k_group * block_k
        k_end = min(k_start + block_k, K)
        scale = As_ref[:, k_group : k_group + 1]  # [M, 1]
        A_scaled[:, k_start:k_end] = A_ref[:, k_start:k_end] * scale

    B_scaled = torch.zeros_like(B_ref)
    for n_group in range(num_n_groups):
        n_start = n_group * block_n
        n_end = min(n_start + block_n, N)
        for k_group in range(num_k_groups):
            k_start = k_group * block_k
            k_end = min(k_start + block_k, K)
            scale = Bs_ref[n_group, k_group]  # scalar
            B_scaled[n_start:n_end, k_start:k_end] = (
                B_ref[n_start:n_end, k_start:k_end] * scale
            )

    ref_out = torch.matmul(A_scaled, B_scaled.T)
    ref_out = utils.to_reference(ref_out, True)
    with flag_gems.use_gems():
        res_out = flag_gems.w8a8_block_fp8_matmul(
            A, B, As, Bs, block_size, output_dtype=torch.float16
        )
    ref_out_fp16 = ref_out.to(torch.float16)
    utils.gems_assert_close(res_out, ref_out_fp16, dtype=torch.float16, reduce_dim=K)
