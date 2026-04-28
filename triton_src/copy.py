import triton
import triton.language as tl


@triton.jit
def copy_kernel_linear(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    # volatile=True bypasses the Triton software cache (add_tritonxpu_memory_cache_pass)
    # on KunlunXin XPU.  Without it, H2D DMA writes (which go directly to HBM) are
    # not visible to the kernel because the on-chip SRAM software cache serves stale
    # data from a previous allocation of the same address.  volatile=True forces a
    # direct HBM read every time, ensuring coherency after H2D transfers.
    # On CUDA this generates ld.volatile which bypasses L1; no measurable cost for a
    # streaming single-pass copy kernel.
    vals = tl.load(src_ptr + offs, mask=mask, volatile=True)
    tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def copy_kernel_nd(
    src_ptr,
    dst_ptr,
    shape_ptr,
    src_stride_ptr,
    dst_stride_ptr,
    numel,
    NDIMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    linear = offs.to(tl.int64)

    src_offset = tl.zeros([BLOCK], dtype=tl.int64)
    dst_offset = tl.zeros([BLOCK], dtype=tl.int64)

    for d in range(NDIMS - 1, -1, -1):
        dim = tl.load(shape_ptr + d, volatile=True)
        idx = linear % dim
        linear = linear // dim
        src_stride = tl.load(src_stride_ptr + d, volatile=True)
        dst_stride = tl.load(dst_stride_ptr + d, volatile=True)
        src_offset += idx * src_stride
        dst_offset += idx * dst_stride

    val = tl.load(src_ptr + src_offset, mask=mask, volatile=True)
    tl.store(dst_ptr + dst_offset, val, mask=mask)
