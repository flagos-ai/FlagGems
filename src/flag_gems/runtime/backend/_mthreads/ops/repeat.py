import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
    ],
    key=["out_shape0", "out_shape1"],
)
@triton.jit
def repeat_kernel_2d(
    inp_ptr,
    out_ptr,
    inp_stride0,
    inp_stride1,
    out_stride0,
    out_stride1,
    inp_shape0,
    inp_shape1,
    out_shape0,
    out_shape1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < out_shape0
    mask_n = offs_n < out_shape1
    mask = mask_m[:, None] & mask_n[None, :]

    # Map output indices to input indices using modulo
    inp_offs_m = offs_m % inp_shape0
    inp_offs_n = offs_n % inp_shape1

    # Load from input
    inp_ptrs = (
        inp_ptr + inp_offs_m[:, None] * inp_stride0 + inp_offs_n[None, :] * inp_stride1
    )
    data = tl.load(inp_ptrs, mask=mask, other=0.0)

    # Store to output
    out_ptrs = out_ptr + offs_m[:, None] * out_stride0 + offs_n[None, :] * out_stride1
    tl.store(out_ptrs, data, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["out_shape0"],
)
@triton.jit
def repeat_kernel_1d(
    inp_ptr,
    out_ptr,
    inp_stride0,
    out_stride0,
    inp_shape0,
    out_shape0,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < out_shape0

    # Map output indices to input indices
    inp_offs = offs % inp_shape0

    # Load and store
    data = tl.load(inp_ptr + inp_offs * inp_stride0, mask=mask)
    tl.store(out_ptr + offs * out_stride0, data, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4),
    ],
    key=["out_shape1", "out_shape2"],
)
@triton.jit
def repeat_kernel_3d(
    inp_ptr,
    out_ptr,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    inp_shape0,
    inp_shape1,
    inp_shape2,
    out_shape0,
    out_shape1,
    out_shape2,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Process 3D repeat: one program handles one (m, n_block, k_block)"""
    pid_m = tle.program_id(0)
    pid_nk = tle.program_id(1)

    num_k_blocks = tl.cdiv(out_shape2, BLOCK_K)
    pid_n = pid_nk // num_k_blocks
    pid_k = pid_nk % num_k_blocks

    m_idx = pid_m
    if m_idx >= out_shape0:
        return

    inp_m = m_idx % inp_shape0

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_n = offs_n < out_shape1
    mask_k = offs_k < out_shape2
    mask = mask_n[:, None] & mask_k[None, :]

    inp_n = offs_n % inp_shape1
    inp_k = offs_k % inp_shape2

    inp_ptrs = (
        inp_ptr
        + inp_m * inp_stride0
        + inp_n[:, None] * inp_stride1
        + inp_k[None, :] * inp_stride2
    )
    data = tl.load(inp_ptrs, mask=mask, other=0.0)

    out_ptrs = (
        out_ptr
        + m_idx * out_stride0
        + offs_n[:, None] * out_stride1
        + offs_k[None, :] * out_stride2
    )
    tl.store(out_ptrs, data, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 32, "BLOCK_L": 32}, num_warps=4),
        triton.Config({"BLOCK_K": 64, "BLOCK_L": 32}, num_warps=4),
        triton.Config({"BLOCK_K": 32, "BLOCK_L": 64}, num_warps=4),
        triton.Config({"BLOCK_K": 64, "BLOCK_L": 64}, num_warps=8),
        triton.Config({"BLOCK_K": 128, "BLOCK_L": 32}, num_warps=4),
        triton.Config({"BLOCK_K": 32, "BLOCK_L": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 128, "BLOCK_L": 64}, num_warps=8),
        triton.Config({"BLOCK_K": 64, "BLOCK_L": 128}, num_warps=8),
    ],
    key=["out_shape2", "out_shape3"],
)
@triton.jit
def repeat_kernel_4d(
    inp_ptr,
    out_ptr,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    inp_stride3,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    inp_shape0,
    inp_shape1,
    inp_shape2,
    inp_shape3,
    out_shape0,
    out_shape1,
    out_shape2,
    out_shape3,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Process 4D repeat: one program handles one (m, n, k_block, l_block)"""
    pid_mn = tle.program_id(0)
    pid_kl = tle.program_id(1)

    num_l_blocks = tl.cdiv(out_shape3, BLOCK_L)
    pid_k = pid_kl // num_l_blocks
    pid_l = pid_kl % num_l_blocks

    # Flatten m, n
    m_idx = pid_mn // out_shape1
    n_idx = pid_mn % out_shape1

    if m_idx >= out_shape0:
        return

    inp_m = m_idx % inp_shape0
    inp_n = n_idx % inp_shape1

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)

    mask_k = offs_k < out_shape2
    mask_l = offs_l < out_shape3
    mask = mask_k[:, None] & mask_l[None, :]

    inp_k = offs_k % inp_shape2
    inp_l = offs_l % inp_shape3

    inp_ptrs = (
        inp_ptr
        + inp_m * inp_stride0
        + inp_n * inp_stride1
        + inp_k[:, None] * inp_stride2
        + inp_l[None, :] * inp_stride3
    )
    data = tl.load(inp_ptrs, mask=mask, other=0.0)

    out_ptrs = (
        out_ptr
        + m_idx * out_stride0
        + n_idx * out_stride1
        + offs_k[:, None] * out_stride2
        + offs_l[None, :] * out_stride3
    )
    tl.store(out_ptrs, data, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
    ],
    key=["num_tasks"],
)
@triton.jit
def repeat_kernel_nd_flat(
    inp_ptr,
    out_ptr,
    num_tasks,
    inp_shape0,
    inp_shape1,
    inp_shape2,
    inp_shape3,
    inp_shape4,
    out_shape0,
    out_shape1,
    out_shape2,
    out_shape3,
    out_shape4,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    inp_stride3,
    inp_stride4,
    out_stride0,
    out_stride1,
    out_stride2,
    out_stride3,
    out_stride4,
    rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generic N-D repeat kernel (up to 5D) using flat indexing with modulo"""
    pid = tle.program_id(0)
    num_ctas = tle.num_programs(0)

    for idx in range(pid * BLOCK_SIZE, num_tasks, num_ctas * BLOCK_SIZE):
        offs = idx + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_tasks

        remaining = offs

        # Unroll for up to 5D
        if rank >= 5:
            out_idx4 = remaining % out_shape4
            inp_idx4 = out_idx4 % inp_shape4
            remaining = remaining // out_shape4
        else:
            out_idx4 = tl.zeros_like(offs)
            inp_idx4 = tl.zeros_like(offs)

        if rank >= 4:
            out_idx3 = remaining % out_shape3
            inp_idx3 = out_idx3 % inp_shape3
            remaining = remaining // out_shape3
        else:
            out_idx3 = tl.zeros_like(offs)
            inp_idx3 = tl.zeros_like(offs)

        if rank >= 3:
            out_idx2 = remaining % out_shape2
            inp_idx2 = out_idx2 % inp_shape2
            remaining = remaining // out_shape2
        else:
            out_idx2 = tl.zeros_like(offs)
            inp_idx2 = tl.zeros_like(offs)

        if rank >= 2:
            out_idx1 = remaining % out_shape1
            inp_idx1 = out_idx1 % inp_shape1
            remaining = remaining // out_shape1
        else:
            out_idx1 = tl.zeros_like(offs)
            inp_idx1 = tl.zeros_like(offs)

        out_idx0 = remaining
        inp_idx0 = out_idx0 % inp_shape0

        inp_offset = (
            inp_idx0 * inp_stride0
            + inp_idx1 * inp_stride1
            + inp_idx2 * inp_stride2
            + inp_idx3 * inp_stride3
            + inp_idx4 * inp_stride4
        )
        out_offset = (
            out_idx0 * out_stride0
            + out_idx1 * out_stride1
            + out_idx2 * out_stride2
            + out_idx3 * out_stride3
            + out_idx4 * out_stride4
        )

        data = tl.load(inp_ptr + inp_offset, mask=mask)
        tl.store(out_ptr + out_offset, data, mask=mask)


def repeat(inp: torch.Tensor, sizes) -> torch.Tensor:
    logger.debug("GEMS_MTHREADS REPEAT")

    in0_rank = inp.dim()
    sizes_rank = len(sizes)
    in0_shape = list(inp.shape)
    sizes_shape = list(sizes)

    # Normalize shapes - for repeat, sizes_rank must be >= in0_rank
    assert (
        sizes_rank >= in0_rank
    ), "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor"

    if sizes_rank > in0_rank:
        diff = sizes_rank - in0_rank
        in0_shape = [1] * diff + in0_shape

    # Check for empty and compute output shape
    is_empty = False
    out_shape = []
    for i in range(len(in0_shape)):
        assert (
            sizes_shape[i] >= 0
        ), f"the number of repetitions per dimension out of range (expected to >= 0) but got {sizes_shape[i]}"
        if sizes_shape[i] == 0:
            is_empty = True
        out_shape.append(in0_shape[i] * sizes_shape[i])

    out = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)

    if is_empty:
        return out

    inp = inp.reshape(in0_shape)
    rank = len(out_shape)
    num_tasks = out.numel()

    # Get strides (handle 0-sized dimensions)
    inp_strides = list(inp.stride())
    out_strides = list(out.stride())

    with torch_device_fn.device(inp.device.index):
        if rank == 1:
            # 1D case with autotune
            grid = lambda META: (triton.cdiv(out_shape[0], META["BLOCK_SIZE"]),)
            repeat_kernel_1d[grid](
                inp,
                out,
                inp_strides[0] if inp_strides[0] != 0 else 1,
                out_strides[0] if out_strides[0] != 0 else 1,
                in0_shape[0],
                out_shape[0],
            )
        elif rank == 2:
            # 2D case - use 2D blocking with autotune
            grid = lambda META: (
                triton.cdiv(out_shape[0], META["BLOCK_M"]),
                triton.cdiv(out_shape[1], META["BLOCK_N"]),
            )
            repeat_kernel_2d[grid](
                inp,
                out,
                inp_strides[0],
                inp_strides[1],
                out_strides[0],
                out_strides[1],
                in0_shape[0],
                in0_shape[1],
                out_shape[0],
                out_shape[1],
            )
        elif rank == 3:
            # 3D case
            grid = lambda META: (
                out_shape[0],
                triton.cdiv(out_shape[1], META["BLOCK_N"])
                * triton.cdiv(out_shape[2], META["BLOCK_K"]),
            )
            repeat_kernel_3d[grid](
                inp,
                out,
                inp_strides[0],
                inp_strides[1],
                inp_strides[2],
                out_strides[0],
                out_strides[1],
                out_strides[2],
                in0_shape[0],
                in0_shape[1],
                in0_shape[2],
                out_shape[0],
                out_shape[1],
                out_shape[2],
            )
        elif rank == 4:
            # 4D case - use 2D grid kernel
            num_mn = out_shape[0] * out_shape[1]
            grid = lambda META: (
                num_mn,
                triton.cdiv(out_shape[2], META["BLOCK_K"])
                * triton.cdiv(out_shape[3], META["BLOCK_L"]),
            )
            repeat_kernel_4d[grid](
                inp,
                out,
                inp_strides[0],
                inp_strides[1],
                inp_strides[2],
                inp_strides[3],
                out_strides[0],
                out_strides[1],
                out_strides[2],
                out_strides[3],
                in0_shape[0],
                in0_shape[1],
                in0_shape[2],
                in0_shape[3],
                out_shape[0],
                out_shape[1],
                out_shape[2],
                out_shape[3],
            )
        else:
            # 5D+ case - use generic kernel with autotune
            # Pad shapes and strides to 5D
            in0_shape_padded = list(in0_shape)
            out_shape_padded = list(out_shape)
            inp_strides_padded = list(inp_strides)
            out_strides_padded = list(out_strides)

            while len(in0_shape_padded) < 5:
                in0_shape_padded = [1] + in0_shape_padded
                out_shape_padded = [1] + out_shape_padded
                inp_strides_padded = [0] + inp_strides_padded
                out_strides_padded = [0] + out_strides_padded

            grid = lambda META: (
                min(65535, triton.cdiv(num_tasks, META["BLOCK_SIZE"])),
            )
            repeat_kernel_nd_flat[grid](
                inp,
                out,
                num_tasks,
                in0_shape_padded[0],
                in0_shape_padded[1],
                in0_shape_padded[2],
                in0_shape_padded[3],
                in0_shape_padded[4],
                out_shape_padded[0],
                out_shape_padded[1],
                out_shape_padded[2],
                out_shape_padded[3],
                out_shape_padded[4],
                inp_strides_padded[0],
                inp_strides_padded[1],
                inp_strides_padded[2],
                inp_strides_padded[3],
                inp_strides_padded[4],
                out_strides_padded[0],
                out_strides_padded[1],
                out_strides_padded[2],
                out_strides_padded[3],
                out_strides_padded[4],
                rank=rank,
            )

    return out
