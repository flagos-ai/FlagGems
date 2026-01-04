import torch
import triton
import triton.language as tl


@triton.jit
def _fill_ones_kernel(
    out_ptr,  # *Pointer* to output tensor
    n_elements,  # Total number of elements to write
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,  # Triton dtype for output
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    ones = tl.full((BLOCK_SIZE,), 1, dtype=DTYPE)
    tl.store(out_ptr + offsets, ones, mask=mask)


def _torch_dtype_to_triton_dtype(dtype: torch.dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float64:
        return tl.float64
    if dtype == torch.int8:
        return tl.int8
    if dtype == torch.uint8:
        return tl.uint8
    if dtype == torch.int16:
        return tl.int16
    if dtype == torch.int32:
        return tl.int32
    if dtype == torch.int64:
        return tl.int64
    # Best-effort for bool
    if dtype == torch.bool:
        # Triton represents boolean as int1
        return tl.int1
    raise TypeError(f"Unsupported dtype for Triton kernel: {dtype}")


def _normalize_size_arg(size):
    if isinstance(size, torch.Size):
        return tuple(int(s) for s in size)
    if isinstance(size, (list, tuple)):
        return tuple(int(s) for s in size)
    if isinstance(size, int):
        return (int(size),)
    raise TypeError(f"Invalid size specification: {size}")


def _extract_size_from_args(after_self_pos_args, kwargs):
    # Try positional
    if len(after_self_pos_args) == 1 and isinstance(
        after_self_pos_args[0], (list, tuple, torch.Size, int)
    ):
        return _normalize_size_arg(after_self_pos_args[0])
    if len(after_self_pos_args) >= 1 and all(
        isinstance(x, int) for x in after_self_pos_args
    ):
        return tuple(int(x) for x in after_self_pos_args)
    # Try kwargs
    if "size" in kwargs:
        return _normalize_size_arg(kwargs["size"])
    if "sizes" in kwargs:
        return _normalize_size_arg(kwargs["sizes"])
    raise ValueError(
        "Size must be provided as a list/tuple/torch.Size or as multiple integer positional arguments."
    )


def _launch_fill_ones(out: torch.Tensor):
    if out.numel() == 0:
        return
    if not out.is_cuda or not out.is_contiguous():
        out.fill_(1)
        return
    triton_dtype = _torch_dtype_to_triton_dtype(out.dtype)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fill_ones_kernel[grid](out, n_elements, BLOCK_SIZE=BLOCK_SIZE, DTYPE=triton_dtype)


def new_ones(*args, **kwargs):
    # Expected schema (aten): new_ones(Tensor self, int[] size, *, dtype?, layout?, device?, pin_memory?)
    if len(args) == 0:
        raise TypeError(
            "new_ones expects at least a 'self' tensor as the first argument."
        )
    self = args[0]
    after_self = list(args[1:])
    size = _extract_size_from_args(after_self, kwargs)

    # Resolve dtype/device defaults
    dtype = kwargs.get("dtype", None)
    if dtype is None:
        if isinstance(self, torch.Tensor) and self.dtype is not None:
            dtype = self.dtype
        else:
            dtype = torch.get_default_dtype()

    device = kwargs.get("device", None)
    if device is None:
        if isinstance(self, torch.Tensor):
            device = self.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = torch.empty(size, dtype=dtype, device=device)
    _launch_fill_ones(out)
    return out


def new_ones_out(*args, **kwargs):
    # Expected schema (aten): new_ones.out(Tensor self, int[] size, *, dtype?,
    # layout?, device?, pin_memory?, Tensor(a!) out) -> Tensor(a!)
    if len(args) == 0:
        raise TypeError(
            "new_ones_out expects at least a 'self' tensor as the first argument."
        )
    _ = args[0]  # self parameter (unused but required by schema)  # noqa: F841
    # Determine 'out' tensor (either in kwargs or as the last positional argument)
    out = kwargs.get("out", None)
    pos_after_self = list(args[1:])

    if (
        out is None
        and len(pos_after_self) >= 1
        and isinstance(pos_after_self[-1], torch.Tensor)
    ):
        out = pos_after_self[-1]
        pos_after_self = pos_after_self[:-1]

    if out is None or not isinstance(out, torch.Tensor):
        raise TypeError(
            "new_ones_out requires an 'out' tensor (as keyword 'out' or last positional argument)."
        )

    size = _extract_size_from_args(pos_after_self, kwargs)

    # Validate/align dtype/device if provided
    if (
        "dtype" in kwargs
        and kwargs["dtype"] is not None
        and out.dtype != kwargs["dtype"]
    ):
        raise TypeError(
            f"Provided dtype {kwargs['dtype']} does not match out.dtype {out.dtype}."
        )
    if (
        "device" in kwargs
        and kwargs["device"] is not None
        and torch.device(kwargs["device"]) != out.device
    ):
        raise TypeError(
            f"Provided device {kwargs['device']} does not match out.device {out.device}."
        )

    # Resize out to requested size if needed
    if tuple(out.shape) != tuple(size):
        out.resize_(size)

    _launch_fill_ones(out)
    return out
