import triton
import triton.language as tl


@triton.jit(debug=True)
def index_fill_contiguous_scalar_kernel(
    out,
    index,
    value,
    outer_index_len,
    index_len,
    dim_size,
    inner_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    inner_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < outer_index_len
    index_coord = m_offsets % index_len
    outer_coord = m_offsets // index_len

    raw_index = tl.load(index + index_coord, mask=m_mask, other=0).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    tl.device_assert(valid_index, "index out of bounds", mask=m_mask)
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)

    out_offsets = outer_coord[:, None] * dim_size * inner_size
    out_offsets += normalized_index[:, None] * inner_size
    out_offsets += inner_offsets[None, :]

    store_mask = m_mask[:, None] & (inner_offsets[None, :] < inner_size)
    store_mask = store_mask & valid_index[:, None]
    tl.store(out + out_offsets, value, mask=store_mask)


@triton.jit(debug=True)
def index_fill_contiguous_scalar_inner1_kernel(
    out,
    index,
    value,
    outer_index_len,
    index_len,
    dim_size,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < outer_index_len

    index_coord = offsets % index_len
    outer_coord = offsets // index_len

    raw_index = tl.load(index + index_coord, mask=mask, other=0).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    tl.device_assert(valid_index, "index out of bounds", mask=mask)
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)

    out_offsets = outer_coord * dim_size + normalized_index
    tl.store(out + out_offsets, value, mask=mask & valid_index)


@triton.jit(debug=True)
def index_fill_contiguous_scalar_small_inner_flat_kernel(
    out,
    index,
    value,
    outer_index_len,
    index_len,
    dim_size,
    BLOCK_M: tl.constexpr,
    INNER_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < outer_index_len

    index_coord = offsets % index_len
    outer_coord = offsets // index_len

    raw_index = tl.load(index + index_coord, mask=mask, other=0).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    tl.device_assert(valid_index, "index out of bounds", mask=mask)
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)

    out_offsets = outer_coord * dim_size * INNER_SIZE
    out_offsets += normalized_index * INNER_SIZE
    store_mask = mask & valid_index

    tl.store(out + out_offsets, value, mask=store_mask)
    if INNER_SIZE >= 2:
        tl.store(out + out_offsets + 1, value, mask=store_mask)
    if INNER_SIZE >= 3:
        tl.store(out + out_offsets + 2, value, mask=store_mask)
    if INNER_SIZE >= 4:
        tl.store(out + out_offsets + 3, value, mask=store_mask)


@triton.jit(debug=True)
def index_fill_contiguous_scalar_inner3_element_kernel(
    out,
    index,
    value,
    row_elements,
    dim_size,
    BLOCK: tl.constexpr,
):
    element_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    outer_offset = tl.program_id(1)
    mask = element_offsets < row_elements

    index_offsets = element_offsets // 3
    inner_offsets = element_offsets % 3
    raw_index = tl.load(index + index_offsets, mask=mask, other=0).to(tl.int64)
    valid_index = (raw_index >= -dim_size) & (raw_index < dim_size)
    tl.device_assert(valid_index, "index out of bounds", mask=mask)
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)

    out_offsets = (outer_offset * dim_size + normalized_index) * 3
    out_offsets += inner_offsets
    tl.store(out + out_offsets, value, mask=mask & valid_index)
