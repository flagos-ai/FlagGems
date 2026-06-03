import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_2d_kernel(inp, out, M: tl.constexpr, N: tl.constexpr):
    """2D transpose: contiguous [M, N] -> contiguous [N, M].

    Each program uses tl.arange(0, 1) to generate a proper tensor operation
    (required by the PPL pipeline's analysisTensorMaxShapeDim), with a mask
    for bounds safety.  Single-element load/store — W-stride = 1 both ways.
    """
    off = tl.arange(0, 1)
    i = tl.program_id(0) + off
    j = tl.program_id(1) + off

    mask = (i < M) & (j < N)
    val = tl.load(inp + i * N + j, mask=mask)
    tl.store(out + j * M + i, val, mask=mask)


def _transpose_2d(inp):
    """2D transpose: contiguous [M, N] -> contiguous [N, M].

    inp must be contiguous row-major, stride (N, 1).
    Uses single-element kernel to avoid DMA W-stride overflow.
    """
    M, N = inp.shape
    out = torch.empty(N, M, dtype=inp.dtype, device=inp.device)
    grid = (M, N)
    _transpose_2d_kernel[grid](inp, out, M, N)
    return out


def safe_permute_contiguous(inp, order):
    """Permute + contiguous with DMA W-stride overflow protection.

    The HW DMA W-stride register is limited to 128 / dtype_size elements.
    When the last dimension in `order` would have stride > limit, this
    function uses a two-step permute or a single-element transpose kernel
    to keep each copy kernel's W-stride safe.
    """
    max_dma_stride = 128 // inp.element_size()
    if inp.stride()[order[-1]] <= max_dma_stride:
        return inp.permute(order).contiguous()

    # 2D simple transpose: use single-element kernel directly
    if inp.ndim == 2 and order == [1, 0]:
        return _transpose_2d(inp)

    # Search for a safe dim among order[:-1] (stride <= limit AND shape <= limit)
    stride = inp.stride()
    safe_dim = None
    for d in order[:-1]:
        if stride[d] <= max_dma_stride and inp.shape[d] <= max_dma_stride:
            safe_dim = d
            break

    if safe_dim is not None:
        safe_dim_pos = order.index(safe_dim)
        n = len(order)
        # Step 1: move safe_dim to the end -> W-stride = stride[safe_dim]
        order1 = [d for d in order if d != safe_dim] + [safe_dim]
        x = inp.permute(*order1).contiguous()
        # Step 2: move safe_dim back to its original position in order
        order2 = list(range(safe_dim_pos)) + [n - 1] + list(range(safe_dim_pos, n - 1))
        x = x.permute(*order2).contiguous()
        return x

    # Fallback: no safe dim found, do direct permute+contiguous
    return inp.permute(order).contiguous()


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return safe_permute_contiguous(inp, order)
