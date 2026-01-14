import torch
import triton
import triton.language as tl


@triton.jit
def _select_backward_scatter_kernel(
    grad_out_ptr,  # pointer to grad_output tensor
    out_ptr,  # pointer to grad_input (output) tensor
    n_elements,  # number of elements in grad_output
    rank_out,  # ndim of grad_output (rank_in - 1)
    index_in_dim,  # index selected along `dim` in the forward pass
    stride_dim,  # stride of `out` along the selected dim
    in_strides_no_dim_ptr,  # int64[stride] array of length rank_out for `out` excluding `dim`
    out_strides_ptr,  # int64[stride] array for grad_output
    out_sizes_ptr,  # int64[size] array for grad_output
    BLOCK_SIZE: tl.constexpr,
    MAX_RANK: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # work with int64 indices
    idx_lin = offs.to(tl.int64)

    rem = idx_lin
    offset_in = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    offset_out = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    # decode linear index into multi-index using out_sizes,
    # and accumulate offsets using respective strides
    for d in tl.static_range(0, MAX_RANK):
        is_active = d < rank_out
        size_d = tl.load(out_sizes_ptr + d, mask=is_active, other=1)
        stride_out_d = tl.load(out_strides_ptr + d, mask=is_active, other=0)
        stride_in_d = tl.load(in_strides_no_dim_ptr + d, mask=is_active, other=0)

        # prevent invalid ops when not active by using size_d=1 above
        idx_d = rem % size_d
        rem = rem // size_d

        offset_out += idx_d * stride_out_d
        offset_in += idx_d * stride_in_d

    # add contribution of the selected dim
    offset_in += stride_dim * index_in_dim

    vals = tl.load(grad_out_ptr + offset_out, mask=mask, other=0)
    tl.store(out_ptr + offset_in, vals, mask=mask)


def _normalize_dim(dim: int, rank_in: int) -> int:
    if dim < 0:
        dim += rank_in
    return dim


def _as_tuple_sizes(sizes):
    if isinstance(sizes, torch.Size):
        return tuple(sizes)
    if isinstance(sizes, (list, tuple)):
        return tuple(int(x) for x in sizes)
    raise TypeError("input_sizes must be a torch.Size, list, or tuple of ints")


def _launch_select_backward(
    grad_output: torch.Tensor, out: torch.Tensor, dim: int, index: int
):
    device = grad_output.device
    assert out.device == device, "grad_output and out must be on the same device"
    assert (
        grad_output.dtype == out.dtype
    ), "grad_output and out must have the same dtype"

    rank_in = out.dim()
    dim = _normalize_dim(dim, rank_in)
    # normalize index
    size_dim = out.size(dim)
    if index < 0:
        index += size_dim
    # shape checks
    expected_out_shape = tuple(out.size(i) for i in range(rank_in) if i != dim)
    assert (
        tuple(grad_output.shape) == expected_out_shape
    ), f"grad_output shape {tuple(grad_output.shape)} does not match expected shape {expected_out_shape}"

    # Prepare strides and sizes (int64 on device)
    in_strides = out.stride()
    collapsed_in_strides = tuple(in_strides[:dim]) + tuple(in_strides[dim + 1 :])
    stride_dim = int(in_strides[dim])

    in_strides_no_dim_t = torch.tensor(
        collapsed_in_strides, dtype=torch.int64, device=device
    )
    out_strides_t = torch.tensor(grad_output.stride(), dtype=torch.int64, device=device)
    out_sizes_t = torch.tensor(grad_output.shape, dtype=torch.int64, device=device)

    n_elements = grad_output.numel()
    if n_elements == 0:
        return

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _select_backward_scatter_kernel[grid](
        grad_output,
        out,
        n_elements,
        grad_output.dim(),
        int(index),
        int(stride_dim),
        in_strides_no_dim_t,
        out_strides_t,
        out_sizes_t,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_RANK=16,
    )


def _unpack_select_backward_args(args, kwargs):
    # aten::select_backward(grad_output, input_sizes, dim, index) -> Tensor
    if "grad_output" in kwargs:
        grad_output = kwargs["grad_output"]
    elif len(args) >= 1:
        grad_output = args[0]
    else:
        raise ValueError("Missing argument: grad_output")

    if "input_sizes" in kwargs:
        input_sizes = kwargs["input_sizes"]
    elif len(args) >= 2:
        input_sizes = args[1]
    else:
        raise ValueError("Missing argument: input_sizes")

    if "dim" in kwargs:
        dim = kwargs["dim"]
    elif len(args) >= 3:
        dim = args[2]
    else:
        raise ValueError("Missing argument: dim")

    if "index" in kwargs:
        index = kwargs["index"]
    elif len(args) >= 4:
        index = args[3]
    else:
        raise ValueError("Missing argument: index")

    return grad_output, input_sizes, int(dim), int(index)


def _unpack_select_backward_out_args(args, kwargs):
    # aten::select_backward.out(grad_output, input_sizes, dim, index, out) -> Tensor
    grad_output, input_sizes, dim, index = _unpack_select_backward_args(args, kwargs)
    if "out" in kwargs:
        out = kwargs["out"]
    elif len(args) >= 5:
        out = args[4]
    else:
        raise ValueError("Missing argument: out")
    return grad_output, input_sizes, dim, index, out


def select_backward(*args, **kwargs):
    grad_output, input_sizes, dim, index = _unpack_select_backward_args(args, kwargs)
    input_sizes = _as_tuple_sizes(input_sizes)
    out = torch.zeros(input_sizes, dtype=grad_output.dtype, device=grad_output.device)
    _launch_select_backward(grad_output, out, dim, index)
    return out


def select_backward_out(*args, **kwargs):
    grad_output, input_sizes, dim, index, out = _unpack_select_backward_out_args(
        args, kwargs
    )
    input_sizes = _as_tuple_sizes(input_sizes)
    assert tuple(out.shape) == tuple(input_sizes), "out tensor has incorrect shape"
    assert (
        out.device == grad_output.device
    ), "out and grad_output must be on the same device"
    assert (
        out.dtype == grad_output.dtype
    ), "out and grad_output must have the same dtype"
    out.zero_()
    _launch_select_backward(grad_output, out, dim, index)
    return out
