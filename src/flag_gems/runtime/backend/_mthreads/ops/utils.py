import os

import numpy as np
import torch
import triton
import triton.language as tl


def create_tma_device_descriptor(tensor, block_m, block_n, device):
    assert tensor.dim() == 2, "TMA descriptor only supports 2D tensors"
    TMA_DESCRIPTOR_SIZE = 64
    desc_np = np.empty(TMA_DESCRIPTOR_SIZE, dtype=np.int8)
    shapes = [tensor.shape[0], tensor.shape[1]]
    if not tensor.is_contiguous():
        assert (
            tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
        ), "TMA descriptor only supports contiguous or transposed 2D tensors"
        shapes.reverse()
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        tensor.data_ptr(),
        shapes[0],
        shapes[1],
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
        and (a_dtype in [torch.float16, torch.bfloat16] and a_dtype.itemsize == 2)
        and ((M, N, K) not in [(1, 1, 32), (15, 160, 1024), (495, 5333, 71)])
    )
