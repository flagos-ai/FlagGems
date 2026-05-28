import torch
import triton
import triton.language as tl


@triton.jit
def _transpose_2d_kernel(inp, out, M: tl.constexpr, N: tl.constexpr):
    """2D transpose: contiguous [M, N] -> contiguous [N, M].

    Each program processes a tile of TILE_M x TILE_N elements, using a
    single-element load/store (tl.arange(0, 1)) for each element within
    the tile, so every DMA W-stride = 1.
    """
    TILE_M: tl.constexpr = 16
    TILE_N: tl.constexpr = 16

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off = tl.arange(0, 1)

    for m_off in range(TILE_M):
        for n_off in range(TILE_N):
            m = pid_m * TILE_M + m_off + off
            n = pid_n * TILE_N + n_off + off
            mask = (m < M) & (n < N)
            val = tl.load(inp + m * N + n, mask=mask)
            tl.store(out + n * M + m, val, mask=mask)


def _transpose_2d(inp):
    """2D transpose: contiguous [M, N] -> contiguous [N, M].

    inp must be contiguous row-major, stride (N, 1).
    Uses tiled single-element kernel to avoid DMA W-stride overflow.
    """
    M, N = inp.shape
    out = torch.empty(N, M, dtype=inp.dtype, device=inp.device)
    TILE_M = 16
    TILE_N = 16
    grid = ((M + TILE_M - 1) // TILE_M, (N + TILE_N - 1) // TILE_N)
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

    # Pseudo-nd with unit dims: squeeze them to reach the 2D _transpose_2d
    # path, then unsqueeze back.  This guarantees every DMA w_stride = 1.
    unit_dims = [d for d in range(inp.ndim) if inp.shape[d] == 1]
    if unit_dims:
        # Build mapping: original dim index -> squeezed dim index (-1 if removed)
        squeeze_map = []
        squeezed_idx = 0
        for d in range(inp.ndim):
            if d not in unit_dims:
                squeeze_map.append(squeezed_idx)
                squeezed_idx += 1
            else:
                squeeze_map.append(-1)
        # Adjust order after removing unit dims
        new_order = [squeeze_map[d] for d in order if squeeze_map[d] >= 0]

        # If all dims are unit dims, any permutation is a no-op
        if len(new_order) == 0:
            return inp.contiguous()

        squeezed = inp
        for d in sorted(unit_dims, reverse=True):
            squeezed = squeezed.squeeze(d)

        result = safe_permute_contiguous(squeezed, new_order)

        # Unsqueeze at the permuted positions, not the original positions.
        # After permutation by `order`, original dim d ends up at position
        # order.index(d) in the output.
        for d in sorted(order.index(d) for d in unit_dims):
            result = result.unsqueeze(d)
        return result

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
