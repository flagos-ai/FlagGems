def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    # The copy kernel maps the last tensor dimension stride to the DMA
    # W-stride. The HW stride register is limited to 128 / dtype_size elements.
    # When stride[order[-1]] exceeds this limit, the DMA hangs. Use a two-step
    # permute so each copy kernel always sees a small W-stride.
    dtype_size = inp.element_size()
    max_dma_stride = 128 // dtype_size
    if stride[order[-1]] > max_dma_stride and len(batch_dim) > 0:
        # Pick the batch dim with the smallest stride to place last, so the
        # first contiguous() copy uses a small W-stride.
        safe_dim = batch_dim[-1]
        safe_dim_pos = order.index(safe_dim)
        n = len(order)
        # Step 1: move safe_dim to the end -> W-stride = stride[safe_dim]
        order1 = [d for d in order if d != safe_dim] + [safe_dim]
        x = inp.permute(*order1).contiguous()
        # Step 2: move safe_dim back to its original position in order
        order2 = list(range(safe_dim_pos)) + [n - 1] + list(range(safe_dim_pos, n - 1))
        x = x.permute(*order2).contiguous()
        return x
    return inp.permute(order).contiguous()
