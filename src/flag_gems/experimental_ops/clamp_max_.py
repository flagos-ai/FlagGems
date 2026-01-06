import torch
import triton
import triton.language as tl


@triton.jit
def clamp_max_tensor_kernel(
    x_ptr,
    other_ptr,
    n_elements,
    in_size0,
    in_size1,
    in_size2,
    in_size3,
    in_size4,
    in_size5,
    in_size6,
    in_size7,
    other_size0,
    other_size1,
    other_size2,
    other_size3,
    other_size4,
    other_size5,
    other_size6,
    other_size7,
    in_stride0,
    in_stride1,
    in_stride2,
    in_stride3,
    in_stride4,
    in_stride5,
    in_stride6,
    in_stride7,
    other_stride0,
    other_stride1,
    other_stride2,
    other_stride3,
    other_stride4,
    other_stride5,
    other_stride6,
    other_stride7,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    rem = offsets

    # Initialize offsets in element units
    offset_in = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    offset_other = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # Dimension 7
    c7 = rem % in_size7
    rem = rem // in_size7
    offset_in += c7 * in_stride7
    use_other7 = other_size7 != 1
    o7 = tl.where(use_other7, c7, 0)
    offset_other += o7 * other_stride7

    # Dimension 6
    c6 = rem % in_size6
    rem = rem // in_size6
    offset_in += c6 * in_stride6
    use_other6 = other_size6 != 1
    o6 = tl.where(use_other6, c6, 0)
    offset_other += o6 * other_stride6

    # Dimension 5
    c5 = rem % in_size5
    rem = rem // in_size5
    offset_in += c5 * in_stride5
    use_other5 = other_size5 != 1
    o5 = tl.where(use_other5, c5, 0)
    offset_other += o5 * other_stride5

    # Dimension 4
    c4 = rem % in_size4
    rem = rem // in_size4
    offset_in += c4 * in_stride4
    use_other4 = other_size4 != 1
    o4 = tl.where(use_other4, c4, 0)
    offset_other += o4 * other_stride4

    # Dimension 3
    c3 = rem % in_size3
    rem = rem // in_size3
    offset_in += c3 * in_stride3
    use_other3 = other_size3 != 1
    o3 = tl.where(use_other3, c3, 0)
    offset_other += o3 * other_stride3

    # Dimension 2
    c2 = rem % in_size2
    rem = rem // in_size2
    offset_in += c2 * in_stride2
    use_other2 = other_size2 != 1
    o2 = tl.where(use_other2, c2, 0)
    offset_other += o2 * other_stride2

    # Dimension 1
    c1 = rem % in_size1
    rem = rem // in_size1
    offset_in += c1 * in_stride1
    use_other1 = other_size1 != 1
    o1 = tl.where(use_other1, c1, 0)
    offset_other += o1 * other_stride1

    # Dimension 0
    c0 = rem % in_size0
    rem = rem // in_size0
    offset_in += c0 * in_stride0
    use_other0 = other_size0 != 1
    o0 = tl.where(use_other0, c0, 0)
    offset_other += o0 * other_stride0

    x = tl.load(x_ptr + offset_in, mask=mask)
    y = tl.load(other_ptr + offset_other, mask=mask)
    out = tl.where(x > y, y, x)
    tl.store(x_ptr + offset_in, out, mask=mask)


def _pad_sizes_strides(shape, strides, target_ndim=8):
    pad = target_ndim - len(shape)
    sizes = ([1] * pad) + list(shape)
    s = ([0] * pad) + list(strides)
    return sizes, s


def _check_broadcastable(in_sizes, other_sizes):
    assert len(in_sizes) == len(other_sizes)
    for a, b in zip(in_sizes, other_sizes):
        if not (b == 1 or b == a):
            raise RuntimeError("Shapes are not broadcastable for clamp_max_.Tensor")


def _launch_clamp_max_tensor_kernel(x: torch.Tensor, other: torch.Tensor):
    assert x.is_cuda and other.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype == other.dtype, "Dtypes must match for in-place clamp_max_"
    ndim = max(x.ndim, other.ndim)
    assert ndim <= 8, "Supported up to 8 dimensions"

    in_sizes, in_strides = _pad_sizes_strides(x.shape, x.stride(), 8)
    other_sizes, other_strides = _pad_sizes_strides(other.shape, other.stride(), 8)
    _check_broadcastable(in_sizes, other_sizes)

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    clamp_max_tensor_kernel[grid](
        x,
        other,
        n_elements,
        in_sizes[0],
        in_sizes[1],
        in_sizes[2],
        in_sizes[3],
        in_sizes[4],
        in_sizes[5],
        in_sizes[6],
        in_sizes[7],
        other_sizes[0],
        other_sizes[1],
        other_sizes[2],
        other_sizes[3],
        other_sizes[4],
        other_sizes[5],
        other_sizes[6],
        other_sizes[7],
        in_strides[0],
        in_strides[1],
        in_strides[2],
        in_strides[3],
        in_strides[4],
        in_strides[5],
        in_strides[6],
        in_strides[7],
        other_strides[0],
        other_strides[1],
        other_strides[2],
        other_strides[3],
        other_strides[4],
        other_strides[5],
        other_strides[6],
        other_strides[7],
        BLOCK_SIZE=1024,
    )


def clamp_max_(*args, **kwargs):
    if len(args) >= 2:
        x, max_val = args[0], args[1]
    else:
        x = kwargs.get("self", kwargs.get("input"))
        max_val = kwargs.get("max")
    if x is None or max_val is None:
        raise ValueError("Expected arguments (Tensor self, Scalar max)")
    assert isinstance(x, torch.Tensor), "self must be a torch.Tensor"
    assert x.is_cuda, "Tensor must be on CUDA device"

    # Create a single-element tensor for max and broadcast via strides
    max_tensor = torch.tensor(max_val, dtype=x.dtype, device=x.device)
    _launch_clamp_max_tensor_kernel(x, max_tensor)
    return x


def clamp_max__Tensor(*args, **kwargs):
    if len(args) >= 2:
        x, other = args[0], args[1]
    else:
        x = kwargs.get("self", kwargs.get("input"))
        other = kwargs.get("max")
    if x is None or other is None:
        raise ValueError("Expected arguments (Tensor self, Tensor max)")
    assert isinstance(x, torch.Tensor) and isinstance(
        other, torch.Tensor
    ), "Arguments must be torch.Tensors"
    assert x.is_cuda and other.is_cuda, "Tensors must be on CUDA device"
    if other.dtype != x.dtype:
        other = other.to(dtype=x.dtype)
    _launch_clamp_max_tensor_kernel(x, other)
    return x
