import triton
import triton.language as tl


@triton.jit
def strided_copy_kernel(
    in_ptr,
    out_ptr,
    in_strides_ptr,
    out_strides_ptr,
    shapes_ptr,
    ndim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    MAX_DIMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offsets = offsets.to(tl.int64)
    mask = offsets < n_elements

    remaining_offset = offsets
    in_physical_offset = tl.zeros_like(offsets)
    out_physical_offset = tl.zeros_like(offsets)

    for i in range(MAX_DIMS - 1, -1, -1):
        is_real_dim = i < ndim
        current_shape = tl.load(shapes_ptr + i, mask=is_real_dim, other=1)
        in_stride_val = tl.load(in_strides_ptr + i, mask=is_real_dim, other=0)
        out_stride_val = tl.load(out_strides_ptr + i, mask=is_real_dim, other=0)

        current_index = remaining_offset % current_shape
        remaining_offset = remaining_offset // current_shape

        in_physical_offset += current_index * in_stride_val
        out_physical_offset += current_index * out_stride_val

    x = tl.load(in_ptr + in_physical_offset, mask=mask)
    tl.store(out_ptr + out_physical_offset, x, mask=mask)
