import torch
import triton
import triton.language as tl


@triton.jit
def _mse_elemwise_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    diff = x - y
    sq = diff * diff
    tl.store(out_ptr + offsets, sq, mask=mask)


@triton.jit
def _mse_reduce_kernel(
    x_ptr, y_ptr, acc_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load as float32 for stable accumulation
    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.float32)
    diff = x - y
    sq = diff * diff
    sq = sq * scale
    block_sum = tl.sum(sq, axis=0)
    tl.atomic_add(acc_ptr, block_sum)


def _parse_reduction(reduction):
    # Accept both strings and integers consistent with ATen Reduction enum:
    # 0: 'none', 1: 'mean', 2: 'sum'
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction string: {reduction}")
    # Assume integer
    if reduction in (0, 1, 2):
        return int(reduction)
    raise ValueError(f"Invalid reduction value: {reduction}")


def _ensure_supported_dtype(t: torch.Tensor, op_name="mse_loss"):
    if t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"{op_name} Triton kernel supports float16, bfloat16, and float32 dtypes, got {t.dtype}."
        )


def _launch_mse_elemwise(x, y, out):
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _mse_elemwise_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)


def _launch_mse_reduce(x, y, n_elements, scale):
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    acc = torch.zeros((), device=x.device, dtype=torch.float32)
    _mse_reduce_kernel[grid](x, y, acc, n_elements, float(scale), BLOCK_SIZE=BLOCK_SIZE)
    return acc


def mse_loss(*args, **kwargs):
    # Expected calling pattern: mse_loss(self, target, reduction=Mean)
    if len(args) < 2:
        raise TypeError(
            "mse_loss requires at least 2 positional arguments: (input, target)"
        )
    inp = args[0]
    target = args[1]
    reduction = kwargs.get("reduction", args[2] if len(args) > 2 else 1)
    reduction = _parse_reduction(reduction)

    if not isinstance(inp, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("mse_loss expects tensor inputs")

    if inp.numel() != target.numel():
        raise ValueError(
            "mse_loss: input and target must have the same number of elements"
        )

    if inp.device != target.device:
        raise ValueError("mse_loss: input and target must be on the same device")

    _ensure_supported_dtype(inp, "mse_loss")
    _ensure_supported_dtype(target, "mse_loss")

    x = inp.contiguous()
    y = target.contiguous()

    n_elements = x.numel()

    if reduction == 0:  # 'none'
        out = torch.empty_like(x)
        if not out.is_contiguous():
            # Ensure output is contiguous for Triton; then copy back
            tmp = torch.empty_like(x, memory_format=torch.contiguous_format)
            _launch_mse_elemwise(x, y, tmp)
            out.copy_(tmp)
        else:
            _launch_mse_elemwise(x, y, out)
        return out.reshape_as(inp)

    # sum or mean -> scalar
    if n_elements == 0:
        # Follow a simple convention: return 0 for empty tensors
        zero = torch.zeros((), device=x.device, dtype=inp.dtype)
        return zero

    scale = 1.0 if reduction == 2 else (1.0 / float(n_elements))  # sum or mean
    acc = _launch_mse_reduce(x, y, n_elements, scale)
    result = acc.to(dtype=inp.dtype)
    return result


def mse_loss_out(*args, **kwargs):
    # Expected calling pattern: mse_loss_out(self, target, reduction=Mean, *, out)
    if len(args) < 2:
        raise TypeError(
            "mse_loss_out requires at least 2 positional arguments: (input, target)"
        )
    inp = args[0]
    target = args[1]
    reduction = kwargs.get("reduction", args[2] if len(args) > 2 else 1)
    out = kwargs.get("out", args[3] if len(args) > 3 else None)

    if out is None:
        raise TypeError("mse_loss_out requires an 'out' tensor")

    reduction = _parse_reduction(reduction)

    if not isinstance(inp, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("mse_loss_out expects tensor inputs")

    if inp.numel() != target.numel():
        raise ValueError(
            "mse_loss_out: input and target must have the same number of elements"
        )

    if inp.device != target.device:
        raise ValueError("mse_loss_out: input and target must be on the same device")

    _ensure_supported_dtype(inp, "mse_loss_out")
    _ensure_supported_dtype(target, "mse_loss_out")

    x = inp.contiguous()
    y = target.contiguous()
    n_elements = x.numel()

    if reduction == 0:  # 'none'
        # out must have same shape as input
        if out.numel() != n_elements:
            raise ValueError(
                "mse_loss_out (reduction='none'): 'out' must have the same number of elements as input"
            )
        if out.device != x.device:
            raise ValueError("mse_loss_out: 'out' must be on the same device as input")
        if out.dtype != inp.dtype:
            raise TypeError(
                "mse_loss_out (reduction='none'): 'out' dtype must match input dtype"
            )

        if out.is_contiguous():
            _launch_mse_elemwise(x, y, out)
        else:
            tmp = torch.empty_like(x, memory_format=torch.contiguous_format)
            _launch_mse_elemwise(x, y, tmp)
            out.copy_(tmp)
        return out

    # sum or mean
    if out.device != x.device:
        raise ValueError("mse_loss_out: 'out' must be on the same device as input")
    if out.numel() != 1:
        raise ValueError(
            "mse_loss_out (reduction in ['sum','mean']): 'out' must be a scalar tensor"
        )
    # out dtype must be a supported float dtype
    if out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "mse_loss_out: 'out' dtype must be one of float16, bfloat16, or float32 for Triton kernel"
        )

    if n_elements == 0:
        out.fill_(0)
        return out

    scale = 1.0 if reduction == 2 else (1.0 / float(n_elements))
    acc = _launch_mse_reduce(x, y, n_elements, scale)
    out.fill_(acc.to(dtype=out.dtype))
    return out
