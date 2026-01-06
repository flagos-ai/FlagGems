import torch
import triton
import triton.language as tl


@triton.jit
def margin_ranking_loss(
    x1_ptr, x2_ptr, target_ptr, out_ptr, n_elements, margin, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0)
    y = tl.load(target_ptr + offsets, mask=mask, other=0)

    diff = x1 - x2
    m = tl.full([BLOCK_SIZE], margin, x1.dtype)
    val = -y * diff + m
    zero = tl.zeros([BLOCK_SIZE], dtype=val.dtype)
    loss = tl.maximum(val, zero)

    tl.store(out_ptr + offsets, loss, mask=mask)


# Preserve a handle to the Triton kernel before defining the Python wrapper with the same name.
_margin_ranking_loss_kernel = margin_ranking_loss


def margin_ranking_loss(*args, **kwargs):
    # Parse inputs: (input1, input2, target, margin=0.0, reduction='mean')
    if len(args) < 3 and not all(k in kwargs for k in ("self", "other", "target")):
        raise TypeError(
            "margin_ranking_loss requires at least three positional arguments: input1, input2, target"
        )

    # Positional extraction
    if len(args) >= 3:
        x1, x2, target = args[0], args[1], args[2]
    else:
        # Fallback to keyword names similar to ATen signature
        x1 = kwargs["self"]
        x2 = kwargs["other"]
        target = kwargs["target"]

    # margin and reduction extraction
    margin = 0.0
    reduction = "mean"
    if len(args) >= 4:
        margin = args[3]
    if len(args) >= 5:
        reduction = args[4]
    if "margin" in kwargs:
        margin = kwargs["margin"]
    if "reduction" in kwargs:
        reduction = kwargs["reduction"]

    # Normalize reduction
    if isinstance(reduction, int):
        reduction = {0: "none", 1: "mean", 2: "sum"}.get(reduction, "mean")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be one of 'none', 'mean', or 'sum'")

    # Device check and fallback
    device = x1.device
    if not (isinstance(device, torch.device) and device.type == "cuda"):
        # Fallback to PyTorch implementation for non-CUDA tensors
        return torch.ops.aten.margin_ranking_loss(
            x1, x2, target, float(margin), {"none": 0, "mean": 1, "sum": 2}[reduction]
        )

    # Broadcast tensors
    x1_b, x2_b, tgt_b = torch.broadcast_tensors(x1, x2, target)

    # Choose dtype (prefer input dtype; fall back to float32 if non-floating)
    common_dtype = x1_b.dtype if x1_b.is_floating_point() else torch.float32
    x1_b = x1_b.to(dtype=common_dtype)
    x2_b = x2_b.to(dtype=common_dtype)
    tgt_b = tgt_b.to(dtype=common_dtype)

    # Flatten contiguous buffers
    x1_c = x1_b.contiguous().view(-1)
    x2_c = x2_b.contiguous().view(-1)
    tgt_c = tgt_b.contiguous().view(-1)

    # Output buffer
    out = torch.empty_like(x1_c)

    n_elements = out.numel()
    if n_elements == 0:
        # Handle empty tensors
        if reduction == "none":
            return out.view(x1_b.shape)
        elif reduction == "sum":
            return out.sum()
        else:
            return out.mean()

    # Launch Triton kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _margin_ranking_loss_kernel[grid](
        x1_c, x2_c, tgt_c, out, n_elements, float(margin), BLOCK_SIZE=BLOCK_SIZE
    )

    # Apply reduction
    if reduction == "none":
        return out.view(x1_b.shape)
    elif reduction == "sum":
        return out.sum()
    else:
        return out.mean()
