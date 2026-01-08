import torch
import triton
import triton.language as tl


@triton.jit
def hinge_embedding_loss(
    input_ptr,  # *Pointer* to input tensor.
    target_ptr,  # *Pointer* to target tensor.
    output_ptr,  # *Pointer* to output tensor (per-element losses).
    n_elements,  # Number of elements.
    margin,  # Margin (float).
    BLOCK_SIZE: tl.constexpr,  # Number of elements per program.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0)
    t = tl.load(target_ptr + offsets, mask=mask, other=0)

    # Cast to float32 for stable math
    x_f = tl.cast(x, tl.float32)
    t_f = tl.cast(t, tl.float32)
    margin_f = tl.full([1], margin, tl.float32)

    # HingeEmbeddingLoss:
    # if target == 1:  loss = input
    # if target == -1: loss = max(0, margin - input)
    # else: 0 (undefined target values)
    is_pos = t_f == 1.0
    is_neg = t_f == -1.0

    loss_pos = x_f
    loss_neg = tl.maximum(0.0, margin_f - x_f)

    loss = tl.where(is_pos, loss_pos, tl.where(is_neg, loss_neg, 0.0))

    # Cast back to input dtype for storage
    loss_out = tl.cast(loss, x.dtype)
    tl.store(output_ptr + offsets, loss_out, mask=mask)


# Preserve kernel reference before defining the Python wrapper with the same name
_hinge_embedding_loss_kernel = hinge_embedding_loss


def hinge_embedding_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    margin: float = 1.0,
    reduction: str | int = "mean",
) -> torch.Tensor:
    # Fallback for non-CUDA tensors
    if input.device.type != "cuda" or target.device.type != "cuda":
        return torch.ops.aten.hinge_embedding_loss(input, target, margin, reduction)

    # Ensure shapes and dtypes
    if input.numel() != target.numel():
        raise ValueError("input and target must have the same number of elements")
    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if input.dtype not in supported_dtypes:
        # Fallback if dtype unsupported by Triton
        return torch.ops.aten.hinge_embedding_loss(input, target, margin, reduction)

    # Allocate output buffer for per-element losses
    out = torch.empty_like(input)

    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _hinge_embedding_loss_kernel[grid](
        input,
        target,
        out,
        n_elements,
        float(margin),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Handle reduction
    red_map = {0: "none", 1: "mean", 2: "sum"}
    if isinstance(reduction, int):
        reduction = red_map.get(reduction, "mean")
    reduction = reduction.lower()

    if reduction == "none":
        return out
    elif reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    else:
        raise ValueError(
            f"Invalid reduction: {reduction}. Supported: 'none', 'mean', 'sum'."
        )
