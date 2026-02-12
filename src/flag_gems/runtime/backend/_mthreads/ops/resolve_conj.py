import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def resolve_conj_kernel_1d(
    x_real_ptr,  # Real part input pointer (float32, separate storage)
    x_img_ptr,  # Imaginary part input pointer (float32, separate storage)
    output_ptr,  # Output pointer (maintain original interleaved layout, float32 view)
    n_elements_total,  # Total number of elements (number of complex pairs)
    is_conj: tl.constexpr,  # Whether to set conjugate flag
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    # Get PID of current program
    pid = tle.program_id(axis=0)

    # Create element index range for current block (complex element index, not float32 index)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask to prevent out-of-bounds access
    mask = offsets < n_elements_total

    # Input: Load real/imaginary parts directly via separate pointers (no need ×2, already separate storage)
    real = tl.load(x_real_ptr + offsets, mask=mask)
    imag = tl.load(x_img_ptr + offsets, mask=mask)

    # Output: Maintain original interleaved layout (real part at even indices, imaginary part at odd indices)
    output_real_offsets = 2 * offsets
    output_img_offsets = 2 * offsets + 1

    if is_conj:
        # Conjugate: Real part unchanged, imaginary part negated, stored in original layout
        tl.store(output_ptr + output_real_offsets, real, mask=mask)
        tl.store(output_ptr + output_img_offsets, -imag, mask=mask)
    else:
        # Direct copy, maintain original layout
        tl.store(output_ptr + output_real_offsets, real, mask=mask)
        tl.store(output_ptr + output_img_offsets, imag, mask=mask)


@triton.jit
def resolve_conj_kernel_2d_strided(
    x_real_ptr,  # Real part input pointer (float32, separate storage)
    x_img_ptr,  # Imaginary part input pointer (float32, separate storage)
    output_ptr,  # Output pointer (maintain original interleaved layout, float32 view)
    n_rows,  # Number of rows
    n_cols,  # Number of columns
    stride_row,  # Row stride (in complex elements)
    stride_col,  # Column stride (in complex elements)
    is_conj: tl.constexpr,  # Whether to set conjugate flag
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    # Get 2D PID of current program
    pid_row = tle.program_id(axis=0)
    pid_col_block = tl.program_id(axis=1)

    # Calculate column index range (complex element index)
    col_start = pid_col_block * BLOCK_SIZE
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE)

    # Create column mask
    col_mask = col_offsets < n_cols

    # Input: Calculate base offset of complex elements (no need ×2, real/imaginary parts separated)
    base_offset = pid_row * stride_row + col_offsets * stride_col

    # Create full mask
    mask = col_mask & (pid_row < n_rows)

    # Load separated real and imaginary parts
    real = tl.load(x_real_ptr + base_offset, mask=mask)
    imag = tl.load(x_img_ptr + base_offset, mask=mask)

    # Output: Convert to interleaved layout offset (×2, real part first, imaginary part second)
    output_base_offset = base_offset * 2

    if is_conj:
        tl.store(output_ptr + output_base_offset, real, mask=mask)
        tl.store(output_ptr + output_base_offset + 1, -imag, mask=mask)
    else:
        tl.store(output_ptr + output_base_offset, real, mask=mask)
        tl.store(output_ptr + output_base_offset + 1, imag, mask=mask)


@triton.jit
def resolve_conj_kernel_large_2d(
    x_real_ptr,  # Real part input pointer (float32, separate storage)
    x_img_ptr,  # Imaginary part input pointer (float32, separate storage)
    output_ptr,  # Output pointer (maintain original interleaved layout, float32 view)
    n_rows,  # Number of rows
    n_cols,  # Number of columns
    stride_row,  # Row stride (in complex elements)
    stride_col,  # Column stride (in complex elements)
    is_conj: tl.constexpr,  # Whether to set conjugate flag
    BLOCK_SIZE_ROWS: tl.constexpr,  # Row block size
    BLOCK_SIZE_COLS: tl.constexpr,  # Column block size
):
    # Get 2D PID of current program
    pid_row = tle.program_id(axis=0)
    pid_col = tle.program_id(axis=1)

    # Calculate row and column index ranges (complex element index)
    row_offsets = pid_row * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    col_offsets = pid_col * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)

    # Create row and column masks
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols

    # Input: Calculate base offset of complex elements (no need ×2, real/imaginary parts separated)
    base_offsets = row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col

    # Create full mask
    mask = row_mask[:, None] & col_mask[None, :]

    # Load separated real and imaginary parts
    real = tl.load(x_real_ptr + base_offsets, mask=mask)
    imag = tl.load(x_img_ptr + base_offsets, mask=mask)

    # Output: Convert to interleaved layout offset (×2)
    output_base_offsets = base_offsets * 2

    if is_conj:
        tl.store(output_ptr + output_base_offsets, real, mask=mask)
        tl.store(output_ptr + output_base_offsets + 1, -imag, mask=mask)
    else:
        tl.store(output_ptr + output_base_offsets, real, mask=mask)
        tl.store(output_ptr + output_base_offsets + 1, imag, mask=mask)


def resolve_conj_triton(x: torch.Tensor, is_conj: bool) -> torch.Tensor:
    """
    resolve_conj function implemented with Triton, supporting arbitrary shapes
    Input: Separate real/imaginary parts (avoid x.view()), Output: Maintain original complex tensor structure

    Args:
        x: Input tensor
        is_conj: Whether conjugate flag is set

    Returns:
        Resolved tensor (structure consistent with input)
    """
    # Ensure tensor is on GPU
    if not x.is_musa:
        x = x.musa()

    # Check if it is complex type
    is_complex = x.is_complex()

    # If no conjugate needed and is real, return copy directly
    if not is_conj and not is_complex:
        return x.clone()

    if not is_complex:
        return x.clone()

    # Output maintains original structure (unchanged), still complex tensor
    output = torch.empty_like(x)

    if x.dtype == torch.complex64:
        # Input separate real/imaginary parts (avoid view(), get float32 tensor directly with .real/.imag)
        x_real = x.real  # shape same as x, dtype=float32 (real part separate storage)
        x_img = (
            x.imag
        )  # shape same as x, dtype=float32 (imaginary part separate storage)

        # Output still use view() to convert to float32 pointer (only for kernel storage, no change to output structure)
        output_view = output.view(torch.float32)

        # Get tensor shape and total number of elements
        shape = x.shape
        n_elements_total = x.numel()

        # Select kernel based on dimensions
        if len(shape) == 2:
            rows, cols = shape

            # Use optimized kernel for large 2D tensors
            if rows * cols > 1000000:
                stride_row = x.stride(0)  # Row stride (complex element unit)
                stride_col = x.stride(1)  # Column stride (complex element unit)

                BLOCK_SIZE_COLS = 128
                grid_rows = rows
                grid_cols = triton.cdiv(cols, BLOCK_SIZE_COLS)
                grid = (grid_rows, grid_cols)

                # Launch kernel (pass separate real/imaginary pointers, output maintains interleaved pointer)
                resolve_conj_kernel_2d_strided[grid](
                    x_real,
                    x_img,
                    output_view,
                    rows,
                    cols,
                    stride_row,
                    stride_col,
                    is_conj,
                    BLOCK_SIZE_COLS,
                )
            else:
                # Use 1D kernel for small 2D tensors
                BLOCK_SIZE = 256
                grid = (triton.cdiv(n_elements_total, BLOCK_SIZE),)
                resolve_conj_kernel_1d[grid](
                    x_real, x_img, output_view, n_elements_total, is_conj, BLOCK_SIZE
                )
        elif len(shape) == 3:
            # Use 1D kernel for 3D tensors (flatten processing)
            n_elements_total = x.numel()
            BLOCK_SIZE = min(1024, n_elements_total)
            grid = (triton.cdiv(n_elements_total, BLOCK_SIZE),)
            resolve_conj_kernel_1d[grid](
                x_real, x_img, output_view, n_elements_total, is_conj, BLOCK_SIZE
            )
        else:
            # Use general 1D kernel for 1D or other dimensions
            BLOCK_SIZE = 1024 if n_elements_total > 1000000 else 256
            grid = (triton.cdiv(n_elements_total, BLOCK_SIZE),)
            resolve_conj_kernel_1d[grid](
                x_real, x_img, output_view, n_elements_total, is_conj, BLOCK_SIZE
            )

        # Output is still complex tensor, structure unchanged
        return output
    else:
        # Unsupported complex type, fallback to PyTorch implementation
        if is_conj:
            return torch.conj(x)
        else:
            return x.clone()


def resolve_conj(A: torch.Tensor):
    logger.debug("GEMS_MTHREADS RESOLVE_CONJ")
    if A.is_conj():
        if len(A.shape) in (2, 3):
            return resolve_conj_triton(A, is_conj=True)
        else:
            return torch.complex(A.real, A.imag.neg())
    else:
        return A
