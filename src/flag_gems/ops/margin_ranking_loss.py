import logging

import torch

import flag_gems

logger = logging.getLogger(__name__)

# @triton.jit
# def _margin_ranking_loss_kernel(
#     x1_ptr, x2_ptr, target_ptr, out_ptr, n_elements, margin, BLOCK_SIZE: tl.constexpr
# ):
#     pid = tl.program_id(axis=0)
#     block_start = pid * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < n_elements

#     x1 = tl.load(x1_ptr + offsets, mask=mask, other=0)
#     x2 = tl.load(x2_ptr + offsets, mask=mask, other=0)
#     y = tl.load(target_ptr + offsets, mask=mask, other=0)

#     diff = x1 - x2
#     m = tl.full([BLOCK_SIZE], margin, x1.dtype)
#     val = -y * diff + m
#     zero = tl.zeros([BLOCK_SIZE], dtype=val.dtype)
#     loss = tl.maximum(val, zero)

#     tl.store(out_ptr + offsets, loss, mask=mask)


def margin_ranking_loss(*args, **kwargs):
    logger.debug("GEMS MARGIN_RANKING_LOSS")
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
    if not (isinstance(device, torch.device) and device.type == flag_gems.device):
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

    # Composite forward: decompose into FlagGems-supported basic ops
    # (sub, mul, neg, add, clamp, mean/sum) so autograd traces through them.
    # When FlagGems is enabled, each of these dispatches to its Triton kernel
    # via the aten override (e.g. aten::sub.Tensor -> FlagGems sub).
    # Formula: loss = clamp(-target * (x1 - x2) + margin, min=0)
    diff = torch.sub(x1_b, x2_b)
    neg_target_diff = torch.mul(torch.neg(tgt_b), diff)
    val = torch.add(neg_target_diff, float(margin))
    loss = torch.clamp(val, min=0.0)

    # Apply reduction
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return torch.sum(loss)
    else:
        return torch.mean(loss)
