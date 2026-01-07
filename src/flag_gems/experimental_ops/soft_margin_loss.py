import torch
import triton
import triton.language as tl


@triton.jit
def _soft_margin_loss_elementwise_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    z = -xf * yf
    absz = tl.abs(z)
    vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))

    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def _soft_margin_loss_sum_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    z = -xf * yf
    absz = tl.abs(z)
    vals = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-absz))
    vals = tl.where(mask, vals, 0.0)

    acc = tl.sum(vals, axis=0)
    tl.atomic_add(out_ptr, acc)


def _normalize_reduction(reduction):
    # Accept both string and enum/int forms: 0=none,1=mean,2=sum
    if isinstance(reduction, str):
        r = reduction.lower()
        if r == "none":
            return 0
        if r == "mean":
            return 1
        if r == "sum":
            return 2
        raise ValueError(f"Invalid reduction: {reduction}")
    if isinstance(reduction, int):
        if reduction in (0, 1, 2):
            return reduction
        raise ValueError(f"Invalid reduction int: {reduction}")
    raise ValueError(f"Unsupported reduction type: {type(reduction)}")


def _check_tensors(input: torch.Tensor, target: torch.Tensor):
    if not (input.is_cuda and target.is_cuda):
        raise AssertionError(
            "soft_margin_loss: input and target must be CUDA tensors for Triton kernel."
        )
    if input.device != target.device:
        raise AssertionError(
            "soft_margin_loss: input and target must be on the same device."
        )
    if input.numel() != target.numel():
        raise AssertionError(
            "soft_margin_loss: input and target must have the same number of elements."
        )
    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    return input, target


def soft_margin_loss(input: torch.Tensor, target: torch.Tensor, reduction="mean"):
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()

    if red == 0:
        # reduction = 'none'
        out = torch.empty_like(input)
        if n_elements == 0:
            return out
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _soft_margin_loss_elementwise_kernel[grid](
            input, target, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    else:
        # reduction = 'sum' or 'mean' (1=mean, 2=sum)
        if n_elements == 0:
            # Follow PyTorch behavior: sum -> 0, mean -> NaN
            if red == 2:
                return torch.zeros((), device=input.device, dtype=input.dtype)
            else:
                return torch.full(
                    (), float("nan"), device=input.device, dtype=input.dtype
                )
        tmp_sum = torch.zeros((), device=input.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _soft_margin_loss_sum_kernel[grid](
            input, target, tmp_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        if red == 2:
            # sum
            return tmp_sum.to(dtype=input.dtype)
        else:
            # mean
            mean_val = (tmp_sum / float(n_elements)).to(dtype=input.dtype)
            return mean_val


def soft_margin_loss_out(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction="mean",
    out: torch.Tensor = None,
):
    input, target = _check_tensors(input, target)
    red = _normalize_reduction(reduction)
    n_elements = input.numel()

    if out is None:
        # Allocate output based on reduction
        if red == 0:
            out = torch.empty_like(input)
        else:
            out = torch.empty((), device=input.device, dtype=input.dtype)
    else:
        if not out.is_cuda:
            raise AssertionError("soft_margin_loss_out: out must be a CUDA tensor.")
        if red == 0:
            if out.numel() != n_elements:
                raise AssertionError(
                    "soft_margin_loss_out: for reduction='none', out must match input shape."
                )
        else:
            if out.numel() != 1:
                raise AssertionError(
                    "soft_margin_loss_out: for reduction='sum' or 'mean', out must be a scalar tensor."
                )
        if out.device != input.device:
            raise AssertionError(
                "soft_margin_loss_out: out must be on the same device as input."
            )

    if red == 0:
        if n_elements > 0:
            BLOCK_SIZE = 1024
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            _soft_margin_loss_elementwise_kernel[grid](
                input, target, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
            )
        return out
    else:
        if n_elements == 0:
            if red == 2:
                out.fill_(0)
            else:
                out.fill_(float("nan"))
            return out
        tmp_sum = torch.zeros((), device=input.device, dtype=torch.float32)
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _soft_margin_loss_sum_kernel[grid](
            input, target, tmp_sum, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        if red == 2:
            out.fill_(tmp_sum.to(dtype=input.dtype))
        else:
            mean_val = (tmp_sum / float(n_elements)).to(dtype=input.dtype)
            out.fill_(mean_val)
        return out
