import torch
import triton
import triton.language as tl


@triton.jit
def logical_xor_(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    a_bool = a != 0
    b_bool = b != 0
    out = a_bool ^ b_bool

    tl.store(a_ptr + offsets, out, mask=mask)


# Preserve reference to the Triton kernel before defining the Python wrapper with the same name.
logical_xor___triton_kernel = logical_xor_


def logical_xor_(*args, **kwargs):
    # Parse inputs: expect (self, other)
    if len(args) >= 2:
        self, other = args[0], args[1]
    else:
        self = kwargs.get("input", kwargs.get("self", None))
        other = kwargs.get("other", None)

    if not isinstance(self, torch.Tensor):
        raise TypeError("logical_xor_: first argument must be a torch.Tensor")
    if self.dtype is not torch.bool:
        raise RuntimeError(
            "logical_xor_: in-place logical operations require self to have dtype torch.bool"
        )

    if not self.is_cuda:
        raise RuntimeError("logical_xor_: tensor must be on CUDA device")

    # Prepare 'other' as tensor on same device
    if isinstance(other, torch.Tensor):
        other_t = other.to(device=self.device)
    else:
        # Create scalar tensor with dtype matching self (bool)
        other_t = torch.tensor(other, device=self.device, dtype=self.dtype)

    # Broadcast 'other' to self's shape and make it contiguous for simple indexing
    try:
        other_bc = torch.broadcast_to(other_t, self.shape).contiguous()
    except Exception as e:
        raise RuntimeError(
            f"logical_xor_: cannot broadcast 'other' to shape {tuple(self.shape)}: {e}"
        )

    # Work on a contiguous copy if self is not contiguous, then copy back
    work_self = self if self.is_contiguous() else self.contiguous()

    n_elements = work_self.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    logical_xor___triton_kernel[grid](work_self, other_bc, n_elements, BLOCK_SIZE=1024)

    if work_self.data_ptr() != self.data_ptr():
        self.copy_(work_self)

    return self
