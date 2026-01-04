import torch
import triton
import triton.language as tl


@triton.jit
def _copy_1d_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)


def _infer_view_size(input_numel, size):
    if isinstance(size, torch.Size):
        size = list(size)
    elif isinstance(size, (list, tuple)):
        size = list(size)
    else:
        raise TypeError("size must be a list/tuple/torch.Size of ints")
    neg_one_count = sum(1 for s in size if s == -1)
    if neg_one_count > 1:
        raise ValueError("only one dimension can be inferred")
    known_prod = 1
    for s in size:
        if s != -1:
            if s < 0:
                raise ValueError("invalid size, negative dimensions other than -1 not allowed")
            known_prod *= s if s != 0 else 1
    if neg_one_count == 0:
        prod = 1
        for s in size:
            prod *= s
        if prod != input_numel:
            raise ValueError(f"requested view size {tuple(size)} does not match input numel {input_numel}")
        return tuple(size)
    else:
        if known_prod == 0:
            if input_numel != 0:
                raise ValueError(f"cannot infer dimension with zero known product and non-zero numel {input_numel}")
            inferred = 0
        else:
            if input_numel % known_prod != 0:
                raise ValueError("input numel not divisible by known product for inferred dimension")
            inferred = input_numel // known_prod
        out = []
        inferred_used = False
        for s in size:
            if s == -1 and not inferred_used:
                out.append(int(inferred))
                inferred_used = True
            else:
                out.append(int(s))
        return tuple(out)


def _launch_copy_kernel(src_flat: torch.Tensor, dst_flat: torch.Tensor):
    assert src_flat.is_cuda and dst_flat.is_cuda, "tensors must be on CUDA device"
    assert src_flat.dtype == dst_flat.dtype, "dtypes must match"
    n_elements = src_flat.numel()
    if n_elements == 0:
        return
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _copy_1d_kernel[grid](src_flat, dst_flat, n_elements, BLOCK_SIZE=1024)


def _unsafe_view(self: torch.Tensor, size):
    new_size = _infer_view_size(self.numel(), size)
    out = torch.empty(new_size, device=self.device, dtype=self.dtype)
    src_flat = self.contiguous().view(-1)
    dst_flat = out.view(-1)
    _launch_copy_kernel(src_flat, dst_flat)
    return out


def _unsafe_view_out(self: torch.Tensor, size, out: torch.Tensor = None):
    if out is None:
        # create out if not provided
        out = torch.empty(0, device=self.device, dtype=self.dtype)
    if out.device != self.device:
        raise ValueError("out tensor must be on the same device as input")
    if out.dtype != self.dtype:
        raise ValueError("out tensor must have the same dtype as input")
    new_size = _infer_view_size(self.numel(), size)
    out.resize_(new_size)
    src_flat = self.contiguous().view(-1)
    dst_flat = out.view(-1)
    _launch_copy_kernel(src_flat, dst_flat)
    return out