import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_warmed_up = set()


@triton.jit
def vecdot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    vec_dim,
    BLOCK_SIZE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    for batch_id in range(pid, batch_size, num_pids):
        batch_offset = batch_id * vec_dim
        acc = tl.zeros((BLOCK_SIZE,), dtype=ACC_DTYPE)
        for start in range(0, vec_dim, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vec_dim
            x = tl.load(x_ptr + batch_offset + offsets, mask=mask, other=0.0).to(
                ACC_DTYPE
            )
            y = tl.load(y_ptr + batch_offset + offsets, mask=mask, other=0.0).to(
                ACC_DTYPE
            )
            acc += x * y
        total = tl.sum(acc, axis=0)
        tl.store(out_ptr + batch_id, total.to(OUTPUT_DTYPE))


def _linalg_vecdot_impl(x, y, dim):
    if dim < 0:
        dim = x.dim() + dim
    vec_dim = x.shape[dim]

    if x.dim() == 1:
        x_flat = x.view(1, vec_dim)
        y_flat = y.view(1, vec_dim)
    else:
        batch_shape = list(x.shape)
        batch_shape.pop(dim)
        if dim == 0:
            x_flat = x.transpose(0, 1).reshape(-1, vec_dim).contiguous()
            y_flat = y.transpose(0, 1).reshape(-1, vec_dim).contiguous()
        else:
            x_flat = x.movedim(dim, -1).reshape(-1, vec_dim).contiguous()
            y_flat = y.movedim(dim, -1).reshape(-1, vec_dim).contiguous()

    batch_size = x_flat.shape[0]
    output = torch.empty(batch_size, dtype=x.dtype, device=x.device)

    if vec_dim <= 32:
        BLOCK_SIZE = 32
    elif vec_dim <= 64:
        BLOCK_SIZE = 64
    elif vec_dim <= 128:
        BLOCK_SIZE = 128
    elif vec_dim <= 256:
        BLOCK_SIZE = 256
    elif vec_dim <= 512:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024

    num_warps = max(1, BLOCK_SIZE // 32)

    if x.dtype == torch.float16:
        acc_dtype = tl.float32
        out_dtype = tl.float16
    elif x.dtype == torch.bfloat16:
        acc_dtype = tl.float32
        out_dtype = tl.bfloat16
    elif x.dtype == torch.float32:
        acc_dtype = tl.float32
        out_dtype = tl.float32
    elif x.dtype == torch.float64:
        acc_dtype = tl.float64
        out_dtype = tl.float64
    else:
        raise ValueError(f"Unsupported dtype {x.dtype}")

    # 最小化 kernel 启动次数
    if batch_size <= 2:
        grid_size = 1
    elif batch_size <= 8:
        grid_size = 2
    elif batch_size <= 16:
        grid_size = 4
    else:
        grid_size = 8
    grid = (grid_size,)

    vecdot_kernel[grid](
        x_flat,
        y_flat,
        output,
        batch_size,
        vec_dim,
        BLOCK_SIZE,
        acc_dtype,
        out_dtype,
        num_warps=num_warps,
    )

    if x.dim() == 1:
        return output.squeeze(0)
    return output.view(batch_shape)


def linalg_vecdot(x, y, dim=-1):
    logger.debug("GEMS LINALG_VECDOT")
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Inputs must be tensors")

    x = x.contiguous()
    y = y.contiguous()
    if x.shape != y.shape:
        raise ValueError("Input shapes must match")

    key = (x.dtype, x.shape, dim)
    if key not in _warmed_up:
        _warmed_up.add(key)
        _linalg_vecdot_impl(x, y, dim)

    return _linalg_vecdot_impl(x, y, dim)
