import triton
import triton.language as tl

MAX_DIM = 8


@triton.jit
def _copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@triton.jit
def _scatter_kernel(
    src_ptr,
    out_ptr,
    SIZES: tl.constexpr,
    ROWSTRIDES: tl.constexpr,
    STRIDES: tl.constexpr,
    SRC_SIZES: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    NDIM: tl.constexpr,
    storage_offset,
    n_view,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    v = pid * BLOCK + tl.arange(0, BLOCK)
    mask = v < n_view
    dst = tl.zeros([BLOCK], dtype=tl.int64) + storage_offset
    src = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(NDIM):
        dim = (v // ROWSTRIDES[d]) % SIZES[d]
        dst += dim.to(tl.int64) * STRIDES[d]
        src += (dim % SRC_SIZES[d]).to(tl.int64) * SRC_STRIDES[d]
    vals = tl.load(src_ptr + src, mask=mask)
    tl.store(out_ptr + dst, vals, mask=mask)


def run(self, src, size, stride, storage_offset=None):
    if storage_offset is None:
        target_offset = self.storage_offset()
    else:
        target_offset = int(storage_offset)
    # Reference semantics: scatter writes to ABSOLUTE storage positions
    # target_offset + sum(idx*stride). Since the returned tensor is `self`
    # (a view at self.storage_offset()), express the offset relative to
    # self.data_ptr() so the kernel writes into the correct storage slots.
    rel_offset = target_offset - self.storage_offset()
    size = [int(x) for x in size]
    stride = [int(x) for x in stride]

    ndim = len(size)
    assert ndim <= MAX_DIM, f"more than {MAX_DIM} dims not supported"
    assert src.dim() <= ndim, "src must be broadcastable to the view size"

    n_view = 1
    for s in size:
        n_view *= s

    n = self.numel()
    if n_view > 0:
        # Fast path: view covers the whole storage range of `self` in
        # canonical row-major order, so the op reduces to out[i] = src[i].
        canon = True
        acc = 1
        for d in range(ndim - 1, -1, -1):
            if stride[d] != acc:
                canon = False
                break
            acc *= size[d]
        is_full = (
            rel_offset == 0
            and canon
            and n_view == n
            and tuple(src.shape) == tuple(size)
            and src.is_contiguous()
        )
        if is_full:
            BLOCK = 1024
            grid = (triton.cdiv(n, BLOCK),)
            _copy_kernel[grid](src, self, n, BLOCK=BLOCK, num_warps=4)
            return self

        pad = MAX_DIM - ndim
        sizes = tuple(size) + (1,) * pad
        strides = tuple(stride) + (0,) * pad
        rowstrides = []
        acc = 1
        for s in reversed(size):
            rowstrides.append(acc)
            acc *= s
        rowstrides = tuple(reversed(rowstrides)) + (1,) * pad

        # src broadcast alignment (trailing dims)
        src_pad = ndim - src.dim()
        src_sizes = (1,) * src_pad + tuple(src.shape)
        src_strides = (0,) * src_pad + tuple(src.stride())
        src_sizes = src_sizes + (1,) * (MAX_DIM - len(src_sizes))
        src_strides = src_strides + (0,) * (MAX_DIM - len(src_strides))

        BLOCK = 1024
        grid = (triton.cdiv(n_view, BLOCK),)
        _scatter_kernel[grid](
            src,
            self,
            sizes,
            rowstrides,
            strides,
            src_sizes,
            src_strides,
            ndim,
            rel_offset,
            n_view,
            BLOCK=BLOCK,
            num_warps=4,
        )

    return self


# Alias for FlagGems import convention
as_strided_scatter = run
