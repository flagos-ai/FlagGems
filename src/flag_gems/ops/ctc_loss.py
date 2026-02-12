import logging

import torch

logger = logging.getLogger(__name__)


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        if reduction == "none":
            return 0
        if reduction == "mean":
            return 1
        if reduction == "sum":
            return 2
        raise ValueError(f"Unsupported reduction: {reduction}")
    return int(reduction)


def _to_int_list(lengths):
    if isinstance(lengths, torch.Tensor):
        return lengths.to("cpu").tolist()
    return list(lengths)


def _ctc_loss_raw(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank,
    zero_infinity,
):
    if isinstance(input_lengths, torch.Tensor) and isinstance(
        target_lengths, torch.Tensor
    ):
        # Move lengths to the same device as log_probs
        device = log_probs.device
        if input_lengths.device != device:
            input_lengths = input_lengths.to(device)
        if target_lengths.device != device:
            target_lengths = target_lengths.to(device)
        return torch.ops.aten._ctc_loss.Tensor(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank,
            zero_infinity,
        )
    input_lengths = _to_int_list(input_lengths)
    target_lengths = _to_int_list(target_lengths)
    return torch.ops.aten._ctc_loss.default(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    )


def _apply_reduction(loss, target_lengths, reduction: int):
    if reduction == 0:
        return loss
    if reduction == 2:
        return loss.sum()
    if reduction == 1:
        if not isinstance(target_lengths, torch.Tensor):
            target_lengths = torch.tensor(
                target_lengths, device=loss.device, dtype=loss.dtype
            )
        else:
            target_lengths = target_lengths.to(device=loss.device, dtype=loss.dtype)
        return (loss / target_lengths).mean()
    raise ValueError(f"Unsupported reduction value: {reduction}")


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank: int = 0,
    reduction: int = 1,
    zero_infinity: bool = False,
):
    logger.debug("GEMS CTC LOSS")
    reduction = _normalize_reduction(reduction)
    loss, _log_alpha = _ctc_loss_raw(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank,
        zero_infinity,
    )
    return _apply_reduction(loss, target_lengths, reduction)
