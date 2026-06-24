import torch
import triton
import triton.language as tl


@triton.jit
def reciprocal_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    one = tl.full([BLOCK_SIZE], 1, x.dtype)
    y = one / x
    tl.store(out_ptr + offsets, y, mask=mask)


def _reciprocal_impl(x: torch.Tensor, out: torch.Tensor = None):
    # Fallback for unsupported dtypes/devices
    if not x.is_cuda or x.is_complex():
        if out is None:
            return torch.ops.aten.reciprocal(x)
        else:
            return torch.ops.aten.reciprocal.out(x, out=out)

    if out is None:
        out = torch.empty_like(x)

    # Ensure same device and dtype
    assert out.device == x.device, "Input and output must be on the same device"
    assert out.dtype == x.dtype, "Output dtype must match input dtype"
    assert (
        out.numel() == x.numel()
    ), "Output must have the same number of elements as input"

    x_contig = x.contiguous()
    out_contig = out.contiguous()

    n_elements = x_contig.numel()
    if n_elements == 0:
        return out  # nothing to do

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    reciprocal_kernel[grid](x_contig, out_contig, n_elements, BLOCK_SIZE=1024)

    if out is not out_contig:
        out.copy_(out_contig)
    return out


# ('reciprocal', <Autograd.disable: False>)
def reciprocal(*args, **kwargs):
    # Accept a single tensor argument
    x = None
    if len(args) >= 1:
        x = args[0]
    else:
        # Try common keyword names
        x = kwargs.get(
            "input", kwargs.get("self", kwargs.get("a", kwargs.get("args", None)))
        )
    if x is None:
        raise ValueError("reciprocal expects a tensor as the first argument")
    return _reciprocal_impl(x)


# ('reciprocal.out', <Autograd.disable: False>)
def reciprocal_out(*args, **kwargs):
    # Accept (x, out) or keyword args self/input and out
    x = None
    out = None
    if len(args) >= 2:
        x, out = args[0], args[1]
    elif len(args) == 1:
        x = args[0]
        out = kwargs.get("out", None)
    else:
        x = kwargs.get("input", kwargs.get("self", kwargs.get("a", None)))
        out = kwargs.get("out", None)

    if x is None or out is None:
        raise ValueError("reciprocal_out expects arguments (input, out)")

    _reciprocal_impl(x, out=out)
    return out
