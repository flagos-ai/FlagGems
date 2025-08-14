import os

import numpy as np
import torch
import triton
import triton.language as tl


def create_tma_device_descriptor(tensor, block_m, block_n, device):
    assert tensor.dim() == 2, "TMA descriptor only supports 2D tensors"
    TMA_DESCRIPTOR_SIZE = 64
    desc_np = np.empty(TMA_DESCRIPTOR_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        tensor.data_ptr(),
        tensor.shape[0],
        tensor.shape[1],
        block_m,
        block_n,
        tensor.element_size(),
        desc_np,
    )
    desc = torch.tensor(desc_np, device=device)
    return desc


def get_triton_dtype(dtype):
    dtype_map = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }
    return dtype_map.get(dtype, None)


def should_enable_sqmma(a_dtype, b_dtype, M, N, K):
    return (
        (os.getenv("MUSA_ENABLE_SQMMA", "0") == "1")
        and (
            ((a_dtype is torch.float16) and (b_dtype is torch.float16))
            or ((a_dtype is torch.bfloat16) and (b_dtype is torch.bfloat16))
        )
        and ((M % 128 == 0) and (N % 128 == 0) and (K % 64 == 0))
    )
