import torch
import triton
import triton.language as tl
import logging

logger = logging.getLogger(__name__)


def _normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


@triton.jit
def diagonal_kernel(
    input_ptr,
    output_ptr,
    in_size0, in_size1, in_size2, in_size3, in_size4,
    in_stride0, in_stride1, in_stride2, in_stride3, in_stride4,
    out_size0, out_size1, out_size2, out_size3, out_size4,
    out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
    ndim: tl.constexpr,
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    offset: tl.constexpr,
    diag_len: tl.constexpr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    idx = offsets

    # ========== 2D ==========
    if ndim == 2:
        # Input: 2D, Output: 1D (diagonal)
        # The output is one-dimensional, corresponding to input dim1=0 and dim2=1
        c0 = idx  # diag index
        if offset >= 0:
            input_offset = c0 * in_stride0 + (c0 + offset) * in_stride1
        else:
            input_offset = (c0 - offset) * in_stride0 + c0 * in_stride1
        val = tl.load(input_ptr + input_offset, mask=mask)
        tl.store(output_ptr + idx, val, mask=mask)

    # ========== 3D ==========
    elif ndim == 3:
        # Output: 2D [out0, out1]
        c1 = idx % out_size1
        c0 = idx // out_size1

        # Compute offset based on dim1/dim2 combination, diagonal is placed at max(dim1, dim2)
        if dim1 == 0 and dim2 == 1:
            # Output shape: [in_size2, diag_len]
            # c0 -> in_size2, c1 -> diag
            if offset >= 0:
                input_offset = c0 * in_stride2 + c1 * in_stride0 + (c1 + offset) * in_stride1
            else:
                input_offset = c0 * in_stride2 + (c1 - offset) * in_stride0 + c1 * in_stride1
        elif dim1 == 0 and dim2 == 2:
            # Output shape: [in_size1, diag_len]
            # c0 -> in_size1, c1 -> diag
            if offset >= 0:
                input_offset = c0 * in_stride1 + c1 * in_stride0 + (c1 + offset) * in_stride2
            else:
                input_offset = c0 * in_stride1 + (c1 - offset) * in_stride0 + c1 * in_stride2
        else:  # dim1 == 1 and dim2 == 2
            # Output shape: [in_size0, diag_len]
            # c0 -> in_size0, c1 -> diag
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride1 + (c1 + offset) * in_stride2
            else:
                input_offset = c0 * in_stride0 + (c1 - offset) * in_stride1 + c1 * in_stride2

        val = tl.load(input_ptr + input_offset, mask=mask)
        tl.store(output_ptr + idx, val, mask=mask)

    # ========== 4D ==========
    elif ndim == 4:
        s1 = out_size1
        s2 = out_size2
        c2 = idx % s2
        c1 = (idx // s2) % s1
        c0 = idx // (s1 * s2)

        # Diagonal is placed at max(dim1, dim2)
        if dim1 == 0 and dim2 == 1:
            # Output shape: [in_size2, in_size3, diag_len]
            # c0 -> in_size2, c1 -> in_size3, c2 -> diag
            if offset >= 0:
                input_offset = c0 * in_stride2 + c1 * in_stride3 + c2 * in_stride0 + (c2 + offset) * in_stride1
            else:
                input_offset = c0 * in_stride2 + c1 * in_stride3 + (c2 - offset) * in_stride0 + c2 * in_stride1
        elif dim1 == 0 and dim2 == 2:
            # Output shape: [in_size1, in_size3, diag_len]
            # c0 -> in_size1, c1 -> in_size3, c2 -> diag
            if offset >= 0:
                input_offset = c0 * in_stride1 + c1 * in_stride3 + c2 * in_stride0 + (c2 + offset) * in_stride2
            else:
                input_offset = c0 * in_stride1 + c1 * in_stride3 + (c2 - offset) * in_stride0 + c2 * in_stride2
        elif dim1 == 0 and dim2 == 3:
            # Output shape: [in_size1, in_size2, diag_len]
            # c0 -> in_size1, c1 -> in_size2, c2 -> diag
            if offset >= 0:
                input_offset = c0 * in_stride1 + c1 * in_stride2 + c2 * in_stride0 + (c2 + offset) * in_stride3
            else:
                input_offset = c0 * in_stride1 + c1 * in_stride2 + (c2 - offset) * in_stride0 + c2 * in_stride3
        elif dim1 == 1 and dim2 == 2:
            # Output shape: [in_size0, in_size3, diag_len]
            # c0 -> in_size0, c1 -> in_size3, c2 -> diag
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride3 + c2 * in_stride1 + (c2 + offset) * in_stride2
            else:
                input_offset = c0 * in_stride0 + c1 * in_stride3 + (c2 - offset) * in_stride1 + c2 * in_stride2
        elif dim1 == 1 and dim2 == 3:
            # Output shape: [in_size0, in_size2, diag_len]
            # c0 -> in_size0, c1 -> in_size2, c2 -> diag
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride2 + c2 * in_stride1 + (c2 + offset) * in_stride3
            else:
                input_offset = c0 * in_stride0 + c1 * in_stride2 + (c2 - offset) * in_stride1 + c2 * in_stride3
        else:  # dim1 == 2 and dim2 == 3
            # Output shape: [in_size0, in_size1, diag_len]
            # c0 -> in_size0, c1 -> in_size1, c2 -> diag
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride1 + c2 * in_stride2 + (c2 + offset) * in_stride3
            else:
                input_offset = c0 * in_stride0 + c1 * in_stride1 + (c2 - offset) * in_stride2 + c2 * in_stride3

        val = tl.load(input_ptr + input_offset, mask=mask)
        tl.store(output_ptr + idx, val, mask=mask)

    # ========== 5D ==========
    elif ndim == 5:
        s1 = out_size1
        s2 = out_size2
        s3 = out_size3
        c3 = idx % s3
        c2 = (idx // s3) % s2
        c1 = (idx // (s2 * s3)) % s1
        c0 = idx // (s1 * s2 * s3)

        if dim1 == 0 and dim2 == 1:
            # Output order: [in2, in3, in4, diag]
            if offset >= 0:
                input_offset = c0 * in_stride2 + c1 * in_stride3 + c2 * in_stride4 + c3 * in_stride0 + (c3 + offset) * in_stride1
            else:
                input_offset = c0 * in_stride2 + c1 * in_stride3 + c2 * in_stride4 + (c3 - offset) * in_stride0 + c3 * in_stride1
        elif dim1 == 0 and dim2 == 2:
            # Output order: [in1, in3, in4, diag]
            if offset >= 0:
                input_offset = c0 * in_stride1 + c1 * in_stride3 + c2 * in_stride4 + c3 * in_stride0 + (c3 + offset) * in_stride2
            else:
                input_offset = c0 * in_stride1 + c1 * in_stride3 + c2 * in_stride4 + (c3 - offset) * in_stride0 + c3 * in_stride2
        elif dim1 == 0 and dim2 == 3:
            # Output order: [in1, in2, in4, diag]
            if offset >= 0:
                input_offset = c0 * in_stride1 + c1 * in_stride2 + c2 * in_stride4 + c3 * in_stride0 + (c3 + offset) * in_stride3
            else:
                input_offset = c0 * in_stride1 + c1 * in_stride2 + c2 * in_stride4 + (c3 - offset) * in_stride0 + c3 * in_stride3
        elif dim1 == 0 and dim2 == 4:
            # Output order: [in1, in2, in3, diag]
            if offset >= 0:
                input_offset = c0 * in_stride1 + c1 * in_stride2 + c2 * in_stride3 + c3 * in_stride0 + (c3 + offset) * in_stride4
            else:
                input_offset = c0 * in_stride1 + c1 * in_stride2 + c2 * in_stride3 + (c3 - offset) * in_stride0 + c3 * in_stride4
        elif dim1 == 1 and dim2 == 2:
            # Output order: [in0, in3, in4, diag]
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride3 + c2 * in_stride4 + c3 * in_stride1 + (c3 + offset) * in_stride2
            else:
                input_offset = c0 * in_stride0 + c1 * in_stride3 + c2 * in_stride4 + (c3 - offset) * in_stride1 + c3 * in_stride2
        elif dim1 == 1 and dim2 == 3:
            # Output order: [in0, in2, in4, diag]
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride2 + c2 * in_stride4 + c3 * in_stride1 + (c3 + offset) * in_stride3
            else:
                input_offset = c0 * in_stride0 + c1 * in_stride2 + c2 * in_stride4 + (c3 - offset) * in_stride1 + c3 * in_stride3
        elif dim1 == 1 and dim2 == 4:
            # Output order: [in0, in2, in3, diag]
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride2 + c2 * in_stride3 + c3 * in_stride1 + (c3 + offset) * in_stride4
            else:
                input_offset = c0 * in_stride0 + c1 * in_stride2 + c2 * in_stride3 + (c3 - offset) * in_stride1 + c3 * in_stride4
        elif dim1 == 2 and dim2 == 3:
            # Output order: [in0, in1, in4, diag]
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride1 + c2 * in_stride4 + c3 * in_stride2 + (c3 + offset) * in_stride3
            else:
                input_offset = c0 * in_stride0 + c1 * in_stride1 + c2 * in_stride4 + (c3 - offset) * in_stride2 + c3 * in_stride3
        elif dim1 == 2 and dim2 == 4:
            # Output order: [in0, in1, in3, diag]
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride1 + c2 * in_stride3 + c3 * in_stride2 + (c3 + offset) * in_stride4
            else:
                input_offset = c0 * in_stride0 + c1 * in_stride1 + c2 * in_stride3 + (c3 - offset) * in_stride2 + c3 * in_stride4
        else:  # dim1 == 3 and dim2 == 4
            # Output order: [in0, in1, in2, diag]
            if offset >= 0:
                input_offset = c0 * in_stride0 + c1 * in_stride1 + c2 * in_stride2 + c3 * in_stride3 + (c3 + offset) * in_stride4
            else:
                input_offset = c0 * in_stride0 + c1 * in_stride1 + c2 * in_stride2 + (c3 - offset) * in_stride3 + c3 * in_stride4

        val = tl.load(input_ptr + input_offset, mask=mask)
        tl.store(output_ptr + idx, val, mask=mask)


def linalg_diagonal(A: torch.Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> torch.Tensor:
    logger.debug("GEMS Triton diagonal")

    if A.dim() < 2:
        raise ValueError("Input tensor must be at least 2-dimensional")

    ndim = A.dim()
    dim1 = _normalize_dim(dim1, ndim)
    dim2 = _normalize_dim(dim2, ndim)
    if dim1 == dim2:
        raise ValueError("dim1 and dim2 cannot be the same")
    # Ensure dim1 < dim2 for simpler handling
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1

    size1 = A.shape[dim1]
    size2 = A.shape[dim2]
    if offset >= 0:
        diag_len = max(0, min(size1, size2 - offset))
    else:
        diag_len = max(0, min(size1 + offset, size2))
    if diag_len == 0:
        out_shape = list(A.shape)
        for d in sorted([dim1, dim2], reverse=True):
            out_shape.pop(d)
        # Insert the diagonal dimension at max(dim1, dim2)
        out_shape.insert(max(dim1, dim2), 0)
        return torch.empty(tuple(out_shape), dtype=A.dtype, device=A.device)

    out_shape = list(A.shape)
    for d in sorted([dim1, dim2], reverse=True):
        out_shape.pop(d)
    # Insert the diagonal dimension at max(dim1, dim2)
    out_shape.insert(max(dim1, dim2), diag_len)
    out_shape = tuple(out_shape)
    out = torch.empty(out_shape, dtype=A.dtype, device=A.device)

    # Pad to 5 dimensions
    in_sizes = list(A.shape) + [1] * (5 - ndim)
    in_strides = list(A.stride()) + [1] * (5 - ndim)
    out_sizes = list(out_shape) + [1] * (5 - len(out_shape))
    out_strides = list(out.stride()) + [1] * (5 - len(out_shape))

    numel = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    diagonal_kernel[grid](
        A, out,
        in_sizes[0], in_sizes[1], in_sizes[2], in_sizes[3], in_sizes[4],
        in_strides[0], in_strides[1], in_strides[2], in_strides[3], in_strides[4],
        out_sizes[0], out_sizes[1], out_sizes[2], out_sizes[3], out_sizes[4],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4],
        ndim=ndim,
        dim1=dim1,
        dim2=dim2,
        offset=offset,
        diag_len=diag_len,
        numel=numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
