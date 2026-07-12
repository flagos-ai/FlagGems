import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


def prepare_tensor_for_kron(tensor_a, tensor_b):
    a_shape = list(tensor_a.shape)
    b_shape = list(tensor_b.shape)

    if tensor_a.numel() == 0 or tensor_b.numel() == 0:
        if not a_shape:
            a_shape = [0]
        if not b_shape:
            b_shape = [0]

        if len(a_shape) > len(b_shape):
            b_shape = [1] * (len(a_shape) - len(b_shape)) + b_shape
        elif len(b_shape) > len(a_shape):
            a_shape = [1] * (len(b_shape) - len(a_shape)) + a_shape

        out_shape = tuple(a * b for a, b in zip(a_shape, b_shape))
        return tensor_a.reshape(*a_shape), tensor_b.reshape(*b_shape), out_shape

    if len(a_shape) < 2:
        a_shape = [1] * (2 - len(a_shape)) + a_shape
    if len(b_shape) < 2:
        b_shape = [1] * (2 - len(b_shape)) + b_shape

    if len(a_shape) > len(b_shape):
        b_shape = [1] * (len(a_shape) - len(b_shape)) + b_shape
    elif len(b_shape) > len(a_shape):
        a_shape = [1] * (len(b_shape) - len(a_shape)) + a_shape

    out_shape = tuple(a * b for a, b in zip(a_shape, b_shape))
    return tensor_a.reshape(*a_shape), tensor_b.reshape(*b_shape), out_shape


def calculate_indices(batch_idx, shape_a, shape_b):
    a_batch_dims = shape_a[:-2] or (1,)
    b_batch_dims = shape_b[:-2] or (1,)
    out_batch_dims = tuple(a * b for a, b in zip(a_batch_dims, b_batch_dims))

    out_indices = []
    remaining = batch_idx
    for dim_size in out_batch_dims[::-1]:
        out_indices.insert(0, remaining % dim_size)
        remaining //= dim_size

    a_idx = b_idx = 0
    for out_idx, (a_dim, b_dim) in zip(out_indices, zip(a_batch_dims, b_batch_dims)):
        a_idx = a_idx * a_dim + (out_idx // b_dim)
        b_idx = b_idx * b_dim + (out_idx % b_dim)

    return a_idx, b_idx


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_TILE_M": 128, "BLOCK_TILE_N": 64}
        ),
        # triton.Config(
        #     {"BLOCK_M": 512, "BLOCK_N": 2048, "BLOCK_TILE_M": 64, "BLOCK_TILE_N": 64}
        # ),
        # triton.Config(
        #     {"BLOCK_M": 512, "BLOCK_N": 8192, "BLOCK_TILE_M": 128, "BLOCK_TILE_N": 64}
        # ),
        # triton.Config(
        #     {"BLOCK_M": 512, "BLOCK_N": 32768, "BLOCK_TILE_M": 128, "BLOCK_TILE_N": 64}
        # ),
    ],
    key=[
        "M",
        "N",
    ],
    warmup=1,
    rep=1,
)
@triton.jit
def kron_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    map_ptr,
    batch_size: tl.int64,
    M: tl.int64,
    N: tl.int64,
    M1: tl.int64,
    M2: tl.int64,
    N1: tl.int64,
    N2: tl.int64,
    a_stride_0: tl.int64,
    a_stride_1: tl.int64,
    b_stride_0: tl.int64,
    b_stride_1: tl.int64,
    c_stride_0: tl.int64,
    c_stride_1: tl.int64,
    a_batch_stride: tl.int64,
    b_batch_stride: tl.int64,
    c_batch_stride: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_TILE_M: tl.constexpr,
    BLOCK_TILE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_per_batch = num_blocks_m * num_blocks_n

    batch_id = pid // num_blocks_per_batch
    local_pid = pid % num_blocks_per_batch
    block_m = local_pid // num_blocks_n
    block_n = local_pid % num_blocks_n

    offset = batch_id * 2
    is_valid_batch = batch_id < batch_size
    a_batch_idx = tl.load(map_ptr + offset, mask=is_valid_batch)
    b_batch_idx = tl.load(map_ptr + offset + 1, mask=is_valid_batch)

    num_tiles_m = tl.cdiv(BLOCK_M, BLOCK_TILE_M)
    num_tiles_n = tl.cdiv(BLOCK_N, BLOCK_TILE_N)

    for tile_m in range(num_tiles_m):
        for tile_n in range(num_tiles_n):
            tile_offset_m = tile_m * BLOCK_TILE_M
            tile_offset_n = tile_n * BLOCK_TILE_N

            current_offs_m = (
                block_m * BLOCK_M + tile_offset_m + tl.arange(0, BLOCK_TILE_M)
            )
            current_offs_n = (
                block_n * BLOCK_N + tile_offset_n + tl.arange(0, BLOCK_TILE_N)
            )

            tile_mask = (
                (current_offs_m[:, None] < M)
                & (current_offs_n[None, :] < N)
                & is_valid_batch
            )

            a_row = current_offs_m[:, None] // M2
            a_col = current_offs_n[None, :] // N2
            b_row = current_offs_m[:, None] % M2
            b_col = current_offs_n[None, :] % N2

            a_idx = (
                a_batch_idx * a_batch_stride + a_row * a_stride_0 + a_col * a_stride_1
            )
            b_idx = (
                b_batch_idx * b_batch_stride + b_row * b_stride_0 + b_col * b_stride_1
            )

            a = tl.load(a_ptr + a_idx, mask=tile_mask)
            b = tl.load(b_ptr + b_idx, mask=tile_mask)
            c = a * b

            c_idx = (
                batch_id * c_batch_stride
                + current_offs_m[:, None] * c_stride_0
                + current_offs_n[None, :] * c_stride_1
            )

            tl.store(c_ptr + c_idx, c, mask=tile_mask)


def kron(A, B):
    logger.debug("GEMS_TSINGMICRO KRON")
    if A.dim() == 0 and B.dim() == 0:
        return A * B

    if A.numel() == 0 or B.numel() == 0:
        A_prepared, B_prepared, out_shape = prepare_tensor_for_kron(A, B)
        output_dtype = torch.promote_types(A.dtype, B.dtype)
        return torch.empty(out_shape, device=A.device, dtype=output_dtype)

    if A.dim() == 0:
        return A.unsqueeze(0) * B
    if B.dim() == 0:
        return A * B.unsqueeze(0)

    A_prepared, B_prepared, out_shape = prepare_tensor_for_kron(A, B)
    M1, N1 = A_prepared.shape[-2:]
    M2, N2 = B_prepared.shape[-2:]
    M, N = M1 * M2, N1 * N2

    batch_size = math.prod(out_shape[:-2]) if out_shape[:-2] else 1

    output_dtype = torch.promote_types(A.dtype, B.dtype)
    C = torch.empty(out_shape, device=A.device, dtype=output_dtype)

    C_reshaped = C.view(-1, M, N)
    A_view = A_prepared.reshape(-1, M1, N1)
    B_view = B_prepared.reshape(-1, M2, N2)

    if not A_view.is_contiguous():
        A_view = A_view.contiguous()
    if not B_view.is_contiguous():
        B_view = B_view.contiguous()

    batch_indices = torch.empty(batch_size * 2, device=A.device, dtype=torch.int64)
    for i in range(batch_size):
        a_idx, b_idx = calculate_indices(i, A_prepared.shape, B_prepared.shape)
        batch_indices[i * 2] = a_idx
        batch_indices[i * 2 + 1] = b_idx

    a_batch_stride = M1 * N1
    b_batch_stride = M2 * N2
    c_batch_stride = M * N

    with torch_device_fn.device(A.device):
        grid = lambda meta: (
            batch_size
            * triton.cdiv(M, meta["BLOCK_M"])
            * triton.cdiv(N, meta["BLOCK_N"]),
        )

        kron_kernel[grid](
            A_view,
            B_view,
            C_reshaped,
            batch_indices,
            batch_size,
            M,
            N,
            M1,
            M2,
            N1,
            N2,
            A_view.stride(1),
            A_view.stride(2),
            B_view.stride(1),
            B_view.stride(2),
            C_reshaped.stride(1),
            C_reshaped.stride(2),
            a_batch_stride,
            b_batch_stride,
            c_batch_stride,
        )

    if A.dim() <= 1 and B.dim() <= 1:
        return C.reshape(-1)

    return C
