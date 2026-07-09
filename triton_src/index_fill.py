import triton
import triton.language as tl


@triton.jit
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
    normalized_index = tl.where(raw_index < 0, raw_index + dim_size, raw_index)

    out_offsets = outer_coord[:, None] * dim_size * inner_size
    out_offsets += normalized_index[:, None] * inner_size
    out_offsets += inner_offsets[None, :]

    store_mask = m_mask[:, None] & (inner_offsets[None, :] < inner_size)
    store_mask = store_mask & valid_index[:, None]
    tl.store(out + out_offsets, value, mask=store_mask)
