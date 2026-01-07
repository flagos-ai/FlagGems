import torch
import triton
import triton.language as tl


@triton.jit
def _masked_scatter_count_kernel(
    mask_ptr,  # *Pointer* to mask tensor (bool)
    counts_ptr,  # *Pointer* to per-block counts (int32)
    n_elements,  # Number of elements in the flattened input
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < n_elements

    m = tl.load(mask_ptr + offsets, mask=in_bounds, other=0)
    m_i32 = m.to(tl.int32)
    local_count = tl.sum(m_i32, axis=0)
    tl.store(counts_ptr + pid, local_count)


@triton.jit
def _masked_scatter_apply_kernel(
    in_ptr,  # *Pointer* to input tensor
    mask_ptr,  # *Pointer* to mask tensor (bool)
    src_ptr,  # *Pointer* to source tensor (1D)
    out_ptr,  # *Pointer* to output tensor
    n_elements,  # Number of elements in the flattened input
    prefix_ptr,  # *Pointer* to per-block exclusive prefix sums (int32)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=in_bounds)
    m = tl.load(mask_ptr + offsets, mask=in_bounds, other=0)
    m_i32 = m.to(tl.int32)

    # Compute per-block exclusive ranks for True mask elements
    inclusive = tl.cumsum(m_i32, axis=0)
    rank = inclusive - m_i32  # exclusive rank within the block

    block_offset = tl.load(prefix_ptr + pid, mask=True, other=0).to(rank.dtype)
    global_rank = block_offset + rank

    take = m_i32 != 0
    gathered = tl.load(src_ptr + global_rank, mask=(in_bounds & take), other=0)

    out_vals = tl.where(take, gathered, x)
    tl.store(out_ptr + offsets, out_vals, mask=in_bounds)


def _launch_masked_scatter(
    input_tensor: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
    out_tensor: torch.Tensor = None,
):
    # Validate inputs
    if input_tensor is None or mask is None or source is None:
        raise ValueError("masked_scatter requires input, mask, and source tensors")

    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    if input_tensor.numel() != mask.numel():
        raise ValueError("input and mask must have the same number of elements")

    if out_tensor is None:
        out = torch.empty_like(input_tensor)
    else:
        out = out_tensor
        if out.shape != input_tensor.shape:
            raise ValueError("out tensor must have the same shape as input")
        if out.dtype != input_tensor.dtype:
            raise ValueError("out tensor must have the same dtype as input")
        if out.device != input_tensor.device:
            raise ValueError("out tensor must be on the same device as input")

    device = input_tensor.device
    if not device.type == "cuda":
        raise ValueError("Triton kernels require CUDA tensors")

    # Flatten to 1D contiguous views
    x_flat = input_tensor.contiguous().view(-1)
    m_flat = mask.contiguous().view(-1)
    s_flat = source.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)

    n_elements = x_flat.numel()
    if n_elements == 0:
        # Nothing to do
        out.copy_(input_tensor)
        return out

    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    # 1) Count number of True mask elements per block
    counts = torch.empty(n_blocks, dtype=torch.int32, device=device)
    grid = (n_blocks,)
    _masked_scatter_count_kernel[grid](
        m_flat, counts, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

    # 2) Compute exclusive prefix sums of per-block counts
    counts_prefix = torch.cumsum(counts, dim=0)
    total_true = int(counts_prefix[-1].item()) if n_blocks > 0 else 0
    if s_flat.numel() < total_true:
        raise ValueError(
            f"source has fewer elements ({s_flat.numel()}) than required by mask ({total_true})"
        )
    prefix_exclusive = counts_prefix - counts  # int32, same device

    # 3) Apply masked_scatter using per-block prefix offsets
    _masked_scatter_apply_kernel[grid](
        x_flat,
        m_flat,
        s_flat,
        out_flat,
        n_elements,
        prefix_exclusive,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape already matches; ensure out has the result
    if out.data_ptr() != out_flat.data_ptr():
        out.view(-1).copy_(out_flat)
    return out


def masked_scatter(input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor):
    return _launch_masked_scatter(input, mask, source, out_tensor=None)


def masked_scatter_out(
    input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor, out: torch.Tensor
):
    return _launch_masked_scatter(input, mask, source, out_tensor=out)
