import torch
import triton
import triton.language as tl

MAX_DIMS = 8
BLOCK_SIZE = 1024


@triton.jit
def maximum_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    s0, s1, s2, s3, s4, s5, s6, s7,         # shape dims
    sa0, sa1, sa2, sa3, sa4, sa5, sa6, sa7, # a strides
    sb0, sb1, sb2, sb3, sb4, sb5, sb6, sb7, # b strides
    so0, so1, so2, so3, so4, so5, so6, so7, # out strides
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Use int64 for address calculations
    li = offsets.to(tl.int64)

    # Compute multi-dimensional indices from linear index (row-major: last dim fastest)
    i7 = li % s7; li = li // s7
    i6 = li % s6; li = li // s6
    i5 = li % s5; li = li // s5
    i4 = li % s4; li = li // s4
    i3 = li % s3; li = li // s3
    i2 = li % s2; li = li // s2
    i1 = li % s1; li = li // s1
    i0 = li % s0; li = li // s0

    # Compute element offsets for each tensor using strides (in elements)
    off_a = (i0 * sa0 + i1 * sa1 + i2 * sa2 + i3 * sa3 +
             i4 * sa4 + i5 * sa5 + i6 * sa6 + i7 * sa7)
    off_b = (i0 * sb0 + i1 * sb1 + i2 * sb2 + i3 * sb3 +
             i4 * sb4 + i5 * sb5 + i6 * sb6 + i7 * sb7)
    off_o = (i0 * so0 + i1 * so1 + i2 * so2 + i3 * so3 +
             i4 * so4 + i5 * so5 + i6 * so6 + i7 * so7)

    a_vals = tl.load(a_ptr + off_a, mask=mask, other=0)
    b_vals = tl.load(b_ptr + off_b, mask=mask, other=0)
    out_vals = tl.maximum(a_vals, b_vals)
    tl.store(out_ptr + off_o, out_vals, mask=mask)


def _as_tensor_on_device(x, device, dtype=None):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype) if (dtype is not None and x.dtype != dtype) or (x.device != device) else x
    return torch.tensor(x, device=device, dtype=dtype)


def _broadcast_to_common(a, b):
    a_b, b_b = torch.broadcast_tensors(a, b)
    return a_b, b_b


def _pad_shape_strides(shape, strides):
    # Ensure shape dims are at least 1 to avoid div by zero
    shape_list = list(shape)
    strides_list = list(strides)
    nd = len(shape_list)
    assert nd <= MAX_DIMS
    shape_list = shape_list + [1] * (MAX_DIMS - nd)
    strides_list = strides_list + [0] * (MAX_DIMS - nd)
    # Triton expects integers
    shape_list = [int(s) for s in shape_list]
    strides_list = [int(s) for s in strides_list]
    return shape_list, strides_list


def _launch_maximum_kernel(a, b, out):
    # Assumes a and b are broadcastable and already cast to out.dtype and on same device
    a_b, b_b = _broadcast_to_common(a, b)
    # Make inputs contiguous to avoid negative/irregular strides complications
    # Broadcasting uses 0-stride for broadcasted dims; keeping 0-stride is fine
    # but handle potential negative/non-standard strides by materializing.
    if any(s < 0 for s in a_b.stride()):
        a_b = a_b.contiguous()
    if any(s < 0 for s in b_b.stride()):
        b_b = b_b.contiguous()

    out_shape = a_b.shape  # == b_b.shape
    n_elements = int(a_b.numel())
    if n_elements == 0:
        return

    # Prepare shape and strides for kernel
    shp, sa = _pad_shape_strides(out_shape, a_b.stride())
    _, sb = _pad_shape_strides(out_shape, b_b.stride())
    _, so = _pad_shape_strides(out_shape, out.stride())

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    maximum_kernel[grid](
        a_b, b_b, out,
        n_elements,
        shp[0], shp[1], shp[2], shp[3], shp[4], shp[5], shp[6], shp[7],
        sa[0], sa[1], sa[2], sa[3], sa[4], sa[5], sa[6], sa[7],
        sb[0], sb[1], sb[2], sb[3], sb[4], sb[5], sb[6], sb[7],
        so[0], so[1], so[2], so[3], so[4], so[5], so[6], so[7],
        BLOCK_SIZE=BLOCK_SIZE,
    )


def maximum(a, b):
    # Determine device
    dev = None
    if torch.is_tensor(a):
        dev = a.device
    if torch.is_tensor(b):
        dev = b.device if dev is None else dev
    if dev is None or dev.type != 'cuda':
        raise ValueError("maximum expects at least one CUDA tensor as input")

    # Determine result dtype per PyTorch promotion rules
    res_dtype = torch.result_type(a, b)
    a_t = _as_tensor_on_device(a, dev, dtype=res_dtype)
    b_t = _as_tensor_on_device(b, dev, dtype=res_dtype)

    # Broadcast to determine output shape
    a_b, b_b = _broadcast_to_common(a_t, b_t)
    out = torch.empty(a_b.shape, device=dev, dtype=res_dtype)

    # If out has negative strides or is non-contiguous, compute into a contiguous buffer then copy
    if not out.is_contiguous() or any(s < 0 for s in out.stride()):
        out_buf = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch_maximum_kernel(a_t, b_t, out_buf)
        out.copy_(out_buf)
    else:
        _launch_maximum_kernel(a_t, b_t, out)

    return out


def maximum_out(a, b, out):
    if not torch.is_tensor(out):
        raise TypeError("out must be a torch.Tensor")
    if out.device.type != 'cuda':
        raise ValueError("out tensor must be on CUDA device")

    dev = out.device

    # Cast inputs to out dtype (following typical .out behavior)
    a_t = _as_tensor_on_device(a, dev, dtype=out.dtype)
    b_t = _as_tensor_on_device(b, dev, dtype=out.dtype)

    # Validate/broadcast shape against out
    a_b, b_b = _broadcast_to_common(a_t, b_t)
    if tuple(a_b.shape) != tuple(out.shape):
        raise ValueError(f"out shape {tuple(out.shape)} is not broadcast-compatible with inputs shape {tuple(a_b.shape)}")

    # If out has negative strides or is non-contiguous, compute into a contiguous buffer then copy
    if not out.is_contiguous() or any(s < 0 for s in out.stride()):
        out_buf = torch.empty_like(out, memory_format=torch.contiguous_format)
        _launch_maximum_kernel(a_t, b_t, out_buf)
        out.copy_(out_buf)
    else:
        _launch_maximum_kernel(a_t, b_t, out)

    return out