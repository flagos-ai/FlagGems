# src/flag_gems/ops/meshgrid.py
from typing import List, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _meshgrid_kernel_2d(
    out0_ptr,
    out1_ptr,
    in0_ptr,
    in1_ptr,
    size0,
    size1,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """2D meshgrid内核 - 向量化版本"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    row_idx = offsets // size1
    col_idx = offsets % size1

    vals0 = tl.load(in0_ptr + row_idx, mask=mask)
    vals1 = tl.load(in1_ptr + col_idx, mask=mask)

    tl.store(out0_ptr + offsets, vals0, mask=mask)
    tl.store(out1_ptr + offsets, vals1, mask=mask)


@triton.jit
def _meshgrid_kernel_2d_tiled(
    out0_ptr,
    out1_ptr,
    in0_ptr,
    in1_ptr,
    size0,
    size1,
    BLOCK_SIZE0: tl.constexpr,
    BLOCK_SIZE1: tl.constexpr,
):
    """
    使用tile的2D meshgrid内核 - 避免除法和取模
    每个块处理一个tile
    """
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # 计算当前块的行和列范围
    row_offsets = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)
    col_offsets = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)

    # 创建掩码
    row_mask = row_offsets < size0
    col_mask = col_offsets < size1

    # 加载输入值（使用向量化）
    in0_vals = tl.load(in0_ptr + row_offsets, mask=row_mask)
    in1_vals = tl.load(in1_ptr + col_offsets, mask=col_mask)

    # ---- 输出0: 每行相同 ----
    # 使用矩阵操作：将行值广播到整个行
    # 计算输出偏移: row * size1 + col
    # 使用tl.broadcast_to实现广播
    out0_vals = tl.broadcast_to(in0_vals[:, None], (BLOCK_SIZE0, BLOCK_SIZE1))
    out1_vals = tl.broadcast_to(in1_vals[None, :], (BLOCK_SIZE0, BLOCK_SIZE1))

    # 计算输出偏移
    row_idx = row_offsets[:, None]
    col_idx = col_offsets[None, :]
    out_offset = row_idx * size1 + col_idx

    # 存储输出
    tl.store(
        out0_ptr + out_offset, out0_vals, mask=row_mask[:, None] & col_mask[None, :]
    )
    tl.store(
        out1_ptr + out_offset, out1_vals, mask=row_mask[:, None] & col_mask[None, :]
    )


@triton.jit
def _meshgrid_kernel_3d(
    out0_ptr,
    out1_ptr,
    out2_ptr,
    in0_ptr,
    in1_ptr,
    in2_ptr,
    size0,
    size1,
    size2,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """3D meshgrid内核"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    size12 = size1 * size2
    idx2 = offsets % size2
    idx1 = (offsets // size2) % size1
    idx0 = offsets // size12

    val0 = tl.load(in0_ptr + idx0, mask=mask)
    tl.store(out0_ptr + offsets, val0, mask=mask)

    val1 = tl.load(in1_ptr + idx1, mask=mask)
    tl.store(out1_ptr + offsets, val1, mask=mask)

    val2 = tl.load(in2_ptr + idx2, mask=mask)
    tl.store(out2_ptr + offsets, val2, mask=mask)


@triton.jit
def _meshgrid_kernel_4d(
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    in0_ptr,
    in1_ptr,
    in2_ptr,
    in3_ptr,
    size0,
    size1,
    size2,
    size3,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """4D meshgrid内核"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    size23 = size2 * size3
    size123 = size1 * size2 * size3
    idx3 = offsets % size3
    idx2 = (offsets // size3) % size2
    idx1 = (offsets // size23) % size1
    idx0 = offsets // size123

    val0 = tl.load(in0_ptr + idx0, mask=mask)
    tl.store(out0_ptr + offsets, val0, mask=mask)

    val1 = tl.load(in1_ptr + idx1, mask=mask)
    tl.store(out1_ptr + offsets, val1, mask=mask)

    val2 = tl.load(in2_ptr + idx2, mask=mask)
    tl.store(out2_ptr + offsets, val2, mask=mask)

    val3 = tl.load(in3_ptr + idx3, mask=mask)
    tl.store(out3_ptr + offsets, val3, mask=mask)


def meshgrid(
    tensors: List[torch.Tensor], indexing: str = "ij"
) -> Tuple[torch.Tensor, ...]:
    """
    模仿 torch.meshgrid 的行为 - 高性能版本
    """
    # ---- 输入验证 ----
    if not tensors:
        raise ValueError("tensors must be a non-empty list or tuple")

    rank = len(tensors)
    if rank > 4:
        raise NotImplementedError("Currently only supports up to 4 dimensions")

    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"tensors[{i}] must be a torch.Tensor")
        if t.dim() != 1:
            raise ValueError(f"tensors[{i}] must be 1D, got shape {t.shape}")
        if not t.is_cuda:
            raise ValueError(f"tensors[{i}] must be on CUDA device")

    if indexing not in ["ij", "xy"]:
        raise ValueError(f"indexing must be 'ij' or 'xy', got {indexing}")

    # ---- 处理输入顺序 ----
    input_tensors = list(tensors)
    if indexing == "xy" and rank >= 2:
        input_tensors[0], input_tensors[1] = input_tensors[1], input_tensors[0]

    # ---- 确定输出形状和元素数 ----
    output_shape = tuple(t.shape[0] for t in input_tensors)
    num_elements = 1
    for dim_size in output_shape:
        num_elements *= dim_size

    # ---- 对于小张量，使用PyTorch的广播 ----
    # 小张量使用PyTorch已经足够快，而且避免了Triton的编译开销
    if num_elements < 200000:
        return torch.meshgrid(*tensors, indexing=indexing)

    # ---- 分配输出张量 ----
    device = input_tensors[0].device
    dtype = input_tensors[0].dtype
    outputs = [
        torch.empty(output_shape, device=device, dtype=dtype) for _ in range(rank)
    ]

    # ---- 根据维度启动内核 ----
    if rank == 2:
        # 对于较大的2D张量，使用优化的tile内核
        if num_elements < 5000000:
            # 中等大小：使用2D tile内核
            BLOCK_SIZE0 = 32
            BLOCK_SIZE1 = 32
            grid_size0 = triton.cdiv(output_shape[0], BLOCK_SIZE0)
            grid_size1 = triton.cdiv(output_shape[1], BLOCK_SIZE1)
            grid = (grid_size0, grid_size1)

            _meshgrid_kernel_2d_tiled[grid](
                outputs[0],
                outputs[1],
                input_tensors[0],
                input_tensors[1],
                output_shape[0],
                output_shape[1],
                BLOCK_SIZE0=BLOCK_SIZE0,
                BLOCK_SIZE1=BLOCK_SIZE1,
            )
        else:
            # 超大：使用1D grid
            BLOCK_SIZE = 512
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

            _meshgrid_kernel_2d[grid](
                outputs[0],
                outputs[1],
                input_tensors[0],
                input_tensors[1],
                output_shape[0],
                output_shape[1],
                num_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
    elif rank == 3:
        # 3D
        if num_elements < 10000000:
            BLOCK_SIZE = 256
        else:
            BLOCK_SIZE = 512
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

        _meshgrid_kernel_3d[grid](
            outputs[0],
            outputs[1],
            outputs[2],
            input_tensors[0],
            input_tensors[1],
            input_tensors[2],
            output_shape[0],
            output_shape[1],
            output_shape[2],
            num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif rank == 4:
        # 4D
        if num_elements < 10000000:
            BLOCK_SIZE = 256
        else:
            BLOCK_SIZE = 512
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

        _meshgrid_kernel_4d[grid](
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            input_tensors[0],
            input_tensors[1],
            input_tensors[2],
            input_tensors[3],
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
            num_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # ---- 处理xy模式的输出交换 ----
    if indexing == "xy" and rank >= 2:
        outputs[0], outputs[1] = outputs[1], outputs[0]

    return tuple(outputs)
