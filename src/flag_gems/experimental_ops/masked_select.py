import torch
import triton
import triton.language as tl


@triton.jit
def _masked_select_count_kernel(
    mask_ptr,  # int32* flattened mask (0/1)
    n_elements,  # int32 number of elements
    counts_ptr,  # int32* per-block counts
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < n_elements

    flags = tl.load(mask_ptr + offsets, mask=in_bounds, other=0)  # int32 0/1
    block_count = tl.sum(flags, axis=0)
    tl.store(counts_ptr + pid, block_count)


@triton.jit
def _masked_select_scatter_kernel(
    input_ptr,  # * input data (flattened, contiguous)
    mask_ptr,  # int32* flattened mask (0/1)
    block_offsets_ptr,  # int32* per-block exclusive offsets
    output_ptr,  # * output data
    n_elements,  # int32 number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < n_elements

    flags = tl.load(mask_ptr + offsets, mask=in_bounds, other=0)  # int32
    # Compute local exclusive positions for true elements
    inclusive = tl.cumsum(flags, axis=0)
    local_pos = inclusive - 1  # valid only where flags == 1

    base = tl.load(block_offsets_ptr + pid)  # int32
    write_idx = base + local_pos

    mstore = in_bounds & (flags != 0)
    vals = tl.load(input_ptr + offsets, mask=mstore, other=0)
    tl.store(output_ptr + write_idx, vals, mask=mstore)


def _prepare_broadcast_flatten(input: torch.Tensor, mask: torch.Tensor):
    # Broadcast input and mask to a common shape
    bshape = torch.broadcast_shapes(tuple(input.shape), tuple(mask.shape))
    inp_b = input.expand(bshape)
    msk_b = mask.to(torch.bool).expand(bshape)

    # Make contiguous flattened views
    inp_flat = inp_b.contiguous().view(-1)
    msk_flat_bool = msk_b.contiguous().view(-1)
    # Convert mask to int32 (0/1) for kernels
    msk_flat_i32 = msk_flat_bool.to(torch.int32)
    return inp_flat, msk_flat_i32


def masked_select(input: torch.Tensor, mask: torch.Tensor):
    inp_flat, msk_flat_i32 = _prepare_broadcast_flatten(input, mask)
    device = inp_flat.device
    assert msk_flat_i32.device == device, "input and mask must be on the same device"

    n_elements = inp_flat.numel()
    if n_elements == 0:
        return torch.empty(0, dtype=input.dtype, device=device)

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    counts = torch.empty(num_blocks, dtype=torch.int32, device=device)
    grid = (num_blocks,)
    _masked_select_count_kernel[grid](
        msk_flat_i32, n_elements, counts, BLOCK_SIZE=BLOCK_SIZE
    )

    # Compute per-block exclusive offsets and total number of selected elements
    if num_blocks == 1:
        block_offsets = torch.zeros(1, dtype=torch.int32, device=device)
        total_selected = int(counts[0].item())
    else:
        prefix = torch.cumsum(counts, dim=0)
        block_offsets = torch.empty_like(counts)
        block_offsets[0] = 0
        block_offsets[1:] = prefix[:-1]
        total_selected = int(prefix[-1].item())

    output = torch.empty(total_selected, dtype=input.dtype, device=device)
    _masked_select_scatter_kernel[grid](
        inp_flat, msk_flat_i32, block_offsets, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return output


def masked_select_out(input: torch.Tensor, mask: torch.Tensor, out: torch.Tensor):
    inp_flat, msk_flat_i32 = _prepare_broadcast_flatten(input, mask)
    device = inp_flat.device
    assert msk_flat_i32.device == device, "input and mask must be on the same device"
    if out.device != device:
        raise RuntimeError("out tensor must be on the same device as input")

    n_elements = inp_flat.numel()
    if n_elements == 0:
        out.resize_(0)
        return out

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    counts = torch.empty(num_blocks, dtype=torch.int32, device=device)
    grid = (num_blocks,)
    _masked_select_count_kernel[grid](
        msk_flat_i32, n_elements, counts, BLOCK_SIZE=BLOCK_SIZE
    )

    # Compute per-block exclusive offsets and total number of selected elements
    if num_blocks == 1:
        block_offsets = torch.zeros(1, dtype=torch.int32, device=device)
        total_selected = int(counts[0].item())
    else:
        prefix = torch.cumsum(counts, dim=0)
        block_offsets = torch.empty_like(counts)
        block_offsets[0] = 0
        block_offsets[1:] = prefix[:-1]
        total_selected = int(prefix[-1].item())

    if out.dtype != input.dtype:
        raise RuntimeError("out tensor dtype must match input dtype")
    out.resize_(total_selected)

    _masked_select_scatter_kernel[grid](
        inp_flat, msk_flat_i32, block_offsets, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out
