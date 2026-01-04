import torch
import triton
import triton.language as tl


@triton.jit
def addcmul_(self_ptr, t1_ptr, t2_ptr, n_elements, value, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(self_ptr + offsets, mask=mask)
    a = tl.load(t1_ptr + offsets, mask=mask)
    b = tl.load(t2_ptr + offsets, mask=mask)

    xf = x.to(tl.float32)
    af = a.to(tl.float32)
    bf = b.to(tl.float32)

    out_f = xf + af * bf * value
    out = out_f.to(x.dtype)

    tl.store(self_ptr + offsets, out, mask=mask)


_addcmul_kernel = addcmul_


def addcmul_(*args, **kwargs):
    # Parse arguments: self, tensor1, tensor2, value (defaults to 1)
    if len(args) == 0:
        raise TypeError("addcmul_ expected at least 1 argument (self tensor)")
    self = args[0]

    # Extract tensor1 and tensor2
    if len(args) >= 3:
        tensor1 = args[1]
        tensor2 = args[2]
        if len(args) >= 4:
            value = args[3]
        else:
            value = kwargs.get("value", kwargs.get("alpha", 1.0))
    else:
        tensor1 = kwargs.get("tensor1", None)
        tensor2 = kwargs.get("tensor2", None)
        value = kwargs.get("value", kwargs.get("alpha", 1.0))

    if tensor1 is None or tensor2 is None:
        raise TypeError("addcmul_ requires tensor1 and tensor2")

    # Convert value to float
    value = float(value)

    # Broadcast tensor1 and tensor2 to match self's shape
    try:
        t1 = tensor1.expand_as(self)
        t2 = tensor2.expand_as(self)
    except Exception:
        t1 = torch.broadcast_to(tensor1, self.shape)
        t2 = torch.broadcast_to(tensor2, self.shape)

    # Fallback conditions
    # - non-CUDA tensors
    # - non-contiguous self (in-place update with non-contiguous memory)
    # - unsupported dtype
    if not (self.is_cuda and t1.is_cuda and t2.is_cuda):
        return torch.ops.aten.addcmul_(self, tensor1, tensor2, value=value)

    if not self.is_contiguous():
        return torch.ops.aten.addcmul_(self, tensor1, tensor2, value=value)

    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return torch.ops.aten.addcmul_(self, tensor1, tensor2, value=value)

    # Make inputs contiguous for efficient loads
    t1 = t1.contiguous()
    t2 = t2.contiguous()

    # Cast inputs to self dtype if needed
    if t1.dtype != self.dtype:
        t1 = t1.to(self.dtype)
    if t2.dtype != self.dtype:
        t2 = t2.to(self.dtype)

    n_elements = self.numel()
    if n_elements == 0:
        return self

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _addcmul_kernel[grid](
        self,
        t1,
        t2,
        n_elements,
        value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return self
