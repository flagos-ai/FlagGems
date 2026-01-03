import torch
import triton
import triton.language as tl


@triton.jit
def _copy_scalar_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, x, mask=mask)


def scalar_tensor(*args, **kwargs):
    # Expected usage: scalar_tensor(value, *, dtype=None, device=None)
    if len(args) == 0 and "value" not in kwargs:
        raise TypeError(
            "scalar_tensor expected a scalar 'value' as the first positional argument or 'value' kwarg"
        )
    value = args[0] if len(args) > 0 else kwargs["value"]
    dtype = kwargs.get("dtype", None)
    device = kwargs.get("device", None)

    # Prepare output 0-d tensor
    if isinstance(value, torch.Tensor) and dtype is None:
        inferred_dtype = value.dtype
    else:
        inferred_dtype = dtype
    out = torch.empty(
        (), dtype=inferred_dtype if inferred_dtype is not None else None, device=device
    )

    # Prepare a 1-element input tensor on the same device/dtype as out
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                "scalar_tensor expects a scalar or 0-d/1-element tensor as input."
            )
        in_buf = value.to(device=out.device, dtype=out.dtype).reshape(1)
    else:
        in_buf = torch.tensor(value, device=out.device, dtype=out.dtype).reshape(1)

    # Launch kernel to copy the single element into the 0-d output
    n_elements = 1
    grid = lambda meta: (1,)
    _copy_scalar_kernel[grid](in_buf, out.view(1), n_elements, BLOCK_SIZE=1)
    return out


def scalar_tensor_out(*args, **kwargs):
    # Expected usage: scalar_tensor.out(value, *, dtype=None, device=None, out=...)
    if len(args) == 0 and "value" not in kwargs:
        raise TypeError(
            "scalar_tensor_out expected a scalar 'value' as the first positional argument or 'value' kwarg"
        )
    value = args[0] if len(args) > 0 else kwargs["value"]

    # 'out' can be provided as kwarg; attempt to also accept as last positional if provided (best-effort)
    out = kwargs.get("out", None)
    if out is None and len(args) > 1:
        out = args[1]
    if out is None:
        raise TypeError("scalar_tensor_out requires an 'out' tensor argument")

    if not isinstance(out, torch.Tensor):
        raise TypeError("'out' must be a torch.Tensor")
    if out.numel() != 1:
        raise ValueError("'out' must be a 0-d (numel==1) tensor")
    # dtype/device in kwargs are ignored for allocation since out is provided; we only cast the input accordingly

    # Prepare a 1-element input tensor on the same device/dtype as out
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                "scalar_tensor_out expects a scalar or 0-d/1-element tensor as input."
            )
        in_buf = value.to(device=out.device, dtype=out.dtype).reshape(1)
    else:
        in_buf = torch.tensor(value, device=out.device, dtype=out.dtype).reshape(1)

    # Launch kernel to copy the single element into the provided 'out'
    n_elements = 1
    grid = lambda meta: (1,)
    _copy_scalar_kernel[grid](in_buf, out.view(1), n_elements, BLOCK_SIZE=1)
    return out
