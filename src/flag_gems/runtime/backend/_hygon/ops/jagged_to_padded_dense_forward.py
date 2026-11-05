import torch
import triton
import triton.language as tl


@triton.jit
def _jagged_to_padded_dense_forward_kernel(
    values_ptr,
    offsets_ptr,
    out_ptr,
    total_elems: tl.constexpr,
    stride: tl.constexpr,  # elements per output row = max_length * D
    inner: tl.constexpr,  # D: elements per value position (1 for 1D values)
    padding_value: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems

    # Map each flat output element to (row, column-within-row).
    row = offs // stride
    col_inner = offs % stride

    seq_start = tl.load(offsets_ptr + row, mask=mask, other=0)
    seq_end = tl.load(offsets_ptr + row + 1, mask=mask, other=0)

    is_data = (col_inner // inner) < (seq_end - seq_start)
    val = tl.load(
        values_ptr + seq_start * inner + col_inner,
        mask=mask & is_data,
        other=padding_value,
    )
    tl.store(out_ptr + offs, val, mask=mask)


def run(values, offsets, max_lengths, padding_value=0.0):
    # --- normalize arguments (harness may pass list-wrapped or raw forms) ---
    if isinstance(offsets, (list, tuple)):
        offsets = offsets[0]
    if isinstance(max_lengths, (list, tuple)):
        max_length = max_lengths[0]
    elif isinstance(max_lengths, torch.Tensor):
        max_length = int(max_lengths.reshape(-1)[0].item())
    else:
        max_length = int(max_lengths)
    if type(padding_value) is not float:
        padding_value = float(
            padding_value.item()
            if isinstance(padding_value, torch.Tensor)
            else padding_value
        )

    batch = offsets.shape[0] - 1
    ndim = values.ndim
    inner = 1 if ndim == 1 else values.shape[1]
    max_len = int(max_length)
    stride = max_len * inner
    total = batch * stride

    if ndim == 1:
        out = torch.empty((batch, max_len), dtype=values.dtype, device=values.device)
    else:
        out = torch.empty(
            (batch, max_len, inner), dtype=values.dtype, device=values.device
        )

    if total == 0:
        return out

    # Adapt BLOCK to the output size: cover tiny outputs with a single program.
    # With total_elems a tl.constexpr and total % BLOCK == 0, Triton drops the
    # bounds mask entirely and can fully vectorize the loads/stores.
    BLOCK = 1024 if total >= 1024 else (1 << (total - 1).bit_length())
    grid = ((total + BLOCK - 1) // BLOCK,)
    num_warps = max(1, min(8, BLOCK // 128))

    _jagged_to_padded_dense_forward_kernel[grid](
        values,
        offsets,
        out,
        total,
        stride=stride,
        inner=inner,
        padding_value=float(padding_value),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out


# Alias for FlagGems import convention
jagged_to_padded_dense_forward = run
