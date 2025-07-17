import triton
import triton.language as tl


@triton.jit
def copy_kernel(
    inp_ptr,
    out_ptr,
    inp_strides_ptr,
    out_strides_ptr,
    shapes_ptr,
    ndim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    offsets = offsets.to(tl.int64)
    mask = offsets < n_elements

    inp_physical_offset = tl.zeros_like(offsets)
    out_physical_offset = tl.zeros_like(offsets)

    for i in range(ndim - 1, -1, -1):
        current_shape = tl.load(shapes_ptr + i, mask=None)
        inp_stride_val = tl.load(inp_strides_ptr + i, mask=None)
        out_stride_val = tl.load(out_strides_ptr + i, mask=None)

        current_index = offsets % current_shape
        offsets = offsets // current_shape

        inp_physical_offset += current_index * inp_stride_val
        out_physical_offset += current_index * out_stride_val

    x = tl.load(inp_ptr + inp_physical_offset, mask)
    tl.store(out_ptr + out_physical_offset, x, mask)
