import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(f'flag_gems.runtime._ascend.ops.{__name__.split(".")[-1]}')

CORE_NUM = 40

try:
    import torch_npu  # noqa: F401
    import triton.runtime.driver as driver

    device = torch.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    CORE_NUM = props["num_vectorcore"]
except Exception:
    CORE_NUM = 40


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
    data_len,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    iter_num = tl.cdiv(BLOCK_SIZE, TILE_SIZE)

    for idx in tl.range(0, iter_num):
        offsets = pid * BLOCK_SIZE + idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = offsets < data_len
        x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
        y = tl.load(y_ptr + offsets, mask=mask, care_padding=False)
        out = x + y * alpha
        tl.store(out_ptr + offsets, out, mask=mask)


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def _launch_add_kernel(x_flat, y_flat, out_flat, alpha, data_len, device):
    TILE_SIZE = 8192
    BLOCK_SIZE = math.ceil(data_len / CORE_NUM)
    BLOCK_SIZE = max(BLOCK_SIZE, TILE_SIZE)
    # Round up to multiple of TILE_SIZE for proper tiling
    BLOCK_SIZE = triton.cdiv(BLOCK_SIZE, TILE_SIZE) * TILE_SIZE
    grid = lambda meta: (triton.cdiv(data_len, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(device):
        add_kernel[grid](
            x_flat, y_flat, out_flat, float(alpha), data_len, BLOCK_SIZE, TILE_SIZE
        )


def add(A, B, *, alpha=1):
    logger.debug("GEMS_ASCEND ADD")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        result_type = torch.result_type(A, B)
        A_cont = A.contiguous()
        B_cont = B.contiguous()
        if A_cont.dtype != result_type:
            A_cont = A_cont.to(result_type)
        if B_cont.dtype != result_type:
            B_cont = B_cont.to(result_type)
        A_flat = A_cont.view(-1)
        B_flat = B_cont.view(-1)
        out = torch.empty_like(A_flat, dtype=result_type)
        data_len = A_flat.numel()
        _launch_add_kernel(A_flat, B_flat, out, alpha, data_len, A.device)
        return out.view(A.shape)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1):
    logger.debug("GEMS_ASCEND ADD_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        A_flat = A.contiguous().view(-1)
        B_flat = B.contiguous().view(-1)
        data_len = A_flat.numel()
        _launch_add_kernel(A_flat, B_flat, A_flat, alpha, data_len, A.device)
        return A
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
