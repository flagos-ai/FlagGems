import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

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


@triton.autotune(configs=runtime.get_tuned_config("kron"), key=["M2", "N2"])
@triton.jit
def kron_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    map_ptr,
    batch_size: tl.int32,
    M1: tl.int32,
    N1: tl.int32,
    M2: tl.int32,
    N2: tl.int32,
    a_stride_0: tl.int32,
    a_stride_1: tl.int32,
    b_stride_0: tl.int32,
    b_stride_1: tl.int32,
    c_stride_0: tl.int32,
    c_stride_1: tl.int32,
    a_batch_stride: tl.int32,
    b_batch_stride: tl.int32,
    c_batch_stride: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: batch × (M1*N1) × cdiv(M2,BLOCK_M) × cdiv(N2,BLOCK_N)
    pid = tle.program_id(0)

    num_b_blocks_n = tl.cdiv(N2, BLOCK_N)
    num_b_blocks_m = tl.cdiv(M2, BLOCK_M)
    num_b_blocks = num_b_blocks_m * num_b_blocks_n
    num_a_elements = M1 * N1
    num_blocks_per_batch = num_a_elements * num_b_blocks

    batch_id = pid // num_blocks_per_batch
    local_pid = pid % num_blocks_per_batch
    a_elem_id = local_pid // num_b_blocks
    b_block_id = local_pid % num_b_blocks

    # A element position (scalar div/rem, not tensor)
    a_row = a_elem_id // N1
    a_col = a_elem_id % N1

    # B tile position (scalar div/rem)
    b_block_m = b_block_id // num_b_blocks_n
    b_block_n = b_block_id % num_b_blocks_n

    b_offs_m = b_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    b_offs_n = b_block_n * BLOCK_N + tl.arange(0, BLOCK_N)

    is_valid = batch_id < batch_size
    b_mask = (b_offs_m[:, None] < M2) & (b_offs_n[None, :] < N2) & is_valid

    # Load batch mapping
    a_batch_idx = tl.load(map_ptr + batch_id * 2, mask=is_valid)
    b_batch_idx = tl.load(map_ptr + batch_id * 2 + 1, mask=is_valid)

    # Load single A element (scalar load)
    a_val = tl.load(
        a_ptr + a_batch_idx * a_batch_stride + a_row * a_stride_0 + a_col * a_stride_1,
        mask=is_valid,
    )

    # Load B tile
    b_idx = (
        b_batch_idx * b_batch_stride
        + b_offs_m[:, None] * b_stride_0
        + b_offs_n[None, :] * b_stride_1
    )
    b_val = tl.load(b_ptr + b_idx, mask=b_mask)

    c = a_val * b_val

    # Output position: only mul + add, no div/rem on tensors
    out_m = a_row * M2 + b_offs_m
    out_n = a_col * N2 + b_offs_n
    c_idx = (
        batch_id * c_batch_stride
        + out_m[:, None] * c_stride_0
        + out_n[None, :] * c_stride_1
    )
    tl.store(c_ptr + c_idx, c, mask=b_mask)


def kron(A, B):
    logging.debug("GEMS KRON SOPHGO")

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

    # Check if exceeds int32 range
    max_int32 = 2147483647
    if batch_size > max_int32 or M > max_int32 or N > max_int32:
        raise RuntimeError(
            f"Tensor dimensions too large for sophgo backend (int32 limit): "
            f"batch_size={batch_size}, M={M}, N={N}"
        )

    output_dtype = torch.promote_types(A.dtype, B.dtype)
    C = torch.empty(out_shape, device=A.device, dtype=output_dtype)

    C_reshaped = C.view(-1, M, N)
    A_view = A_prepared.reshape(-1, M1, N1)
    B_view = B_prepared.reshape(-1, M2, N2)

    if not A_view.is_contiguous():
        A_view = A_view.contiguous()
    if not B_view.is_contiguous():
        B_view = B_view.contiguous()

    # Use int32 instead of int64
    batch_indices = torch.empty(batch_size * 2, device=A.device, dtype=torch.int32)
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
            * (M1 * N1)
            * triton.cdiv(M2, meta["BLOCK_M"])
            * triton.cdiv(N2, meta["BLOCK_N"]),
        )

        kron_kernel[grid](
            A_view,
            B_view,
            C_reshaped,
            batch_indices,
            batch_size,
            M1,
            N1,
            M2,
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
