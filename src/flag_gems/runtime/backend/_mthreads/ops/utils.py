import os

import torch
import triton.language as tl


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
